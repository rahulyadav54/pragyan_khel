from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

app = Flask(__name__)

model = YOLO('yolo11n-seg.pt')
selected_object = None
blur_intensity = 71
camera = None
uploaded_video_path = None
latest_detections = []
low_light_active = False

LOW_LIGHT_THRESHOLD = 90.0
MAX_LOST_FRAMES = 30

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / float(area1 + area2 - inter)

def box_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

def box_diag(box):
    return max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))

def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]

def estimate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def enhance_low_light(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    merged = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    gamma = 1.35
    lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255.0)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, lut)

def prepare_detection_frame(frame):
    brightness = estimate_brightness(frame)
    if brightness < LOW_LIGHT_THRESHOLD:
        return enhance_low_light(frame), True
    return frame, False

def create_mask(frame, detection):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    seg_mask = detection.get('mask')

    if seg_mask is not None:
        if seg_mask.shape != mask.shape:
            seg_mask = cv2.resize(seg_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = (seg_mask > 0.5).astype(np.uint8) * 255
    else:
        x1, y1, x2, y2 = map(int, detection['box'])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    return mask

def apply_blur(frame, mask, intensity):
    blur_size = intensity if intensity % 2 == 1 else intensity + 1
    blur_size = max(blur_size, 5)
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    mask_norm = mask.astype(float) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=2)
    return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

def find_best_match(previous_object, detections):
    prev_box = previous_object['box']
    prev_cls = previous_object['cls']
    prev_velocity = previous_object.get('velocity', [0, 0, 0, 0])
    pred_box = [prev_box[i] + prev_velocity[i] for i in range(4)]

    best_det = None
    best_score = -1.0
    pred_cx, pred_cy = box_center(pred_box)
    pred_diag = box_diag(pred_box)
    for det in detections:
        if det['cls'] != prev_cls:
            continue
        iou = compute_iou(pred_box, det['box'])
        cx, cy = box_center(det['box'])
        center_dist = np.hypot(cx - pred_cx, cy - pred_cy) / pred_diag
        motion_term = max(0.0, 1.0 - (center_dist / 2.8))
        score = (iou * 0.68) + (motion_term * 0.32)
        if score > best_score:
            best_score = score
            best_det = det

    if best_det is not None and best_score >= 0.18:
        return best_det

    prev_cx, prev_cy = box_center(prev_box)
    distance_candidates = []
    for det in detections:
        if det['cls'] != prev_cls:
            continue
        cx, cy = box_center(det['box'])
        dist_sq = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
        distance_candidates.append((dist_sq, det))

    if not distance_candidates:
        return None

    distance_candidates.sort(key=lambda x: x[0])
    return distance_candidates[0][1]

def generate_frames():
    global selected_object, blur_intensity, camera, latest_detections, low_light_active
    
    while True:
        if camera is None:
            break
        success, frame = camera.read()
        if not success:
            break

        detection_frame, low_light_active = prepare_detection_frame(frame)
        results = model(detection_frame, verbose=False)[0]
        detections = []
        seg_masks = results.masks.data.cpu().numpy() if results.masks is not None else None

        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            det = {
                'box': [x1, y1, x2, y2],
                'cls': cls_id,
                'mask': seg_masks[idx] if seg_masks is not None and idx < len(seg_masks) else None
            }
            detections.append(det)
        latest_detections = detections

        if selected_object:
            matched = find_best_match(selected_object, detections)
            if matched is not None:
                old_box = selected_object['box']
                new_box = matched['box']
                raw_velocity = [new_box[i] - old_box[i] for i in range(4)]
                old_velocity = selected_object.get('velocity', [0, 0, 0, 0])
                smooth_velocity = [(old_velocity[i] * 0.6) + (raw_velocity[i] * 0.4) for i in range(4)]
                selected_object = {
                    'box': new_box,
                    'cls': matched['cls'],
                    'mask': matched.get('mask'),
                    'velocity': smooth_velocity,
                    'lost_frames': 0
                }
                mask = create_mask(frame, selected_object)
                frame = apply_blur(frame, mask, blur_intensity)
            else:
                selected_object['lost_frames'] = selected_object.get('lost_frames', 0) + 1
                if selected_object['lost_frames'] <= MAX_LOST_FRAMES:
                    h, w = frame.shape[:2]
                    vel = selected_object.get('velocity', [0, 0, 0, 0])
                    predicted_box = [
                        selected_object['box'][i] + vel[i] for i in range(4)
                    ]
                    predicted_box = clamp_box(predicted_box, w, h)
                    selected_object['box'] = predicted_box
                    selected_object['velocity'] = [v * 0.8 for v in vel]
                    fallback_detection = {
                        'box': predicted_box,
                        'cls': selected_object['cls'],
                        'mask': None
                    }
                    mask = create_mask(frame, fallback_detection)
                    frame = apply_blur(frame, mask, blur_intensity)
                else:
                    selected_object = None
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global camera, selected_object, latest_detections, low_light_active
    if camera:
        camera.release()
    camera = cv2.VideoCapture(0)
    selected_object = None
    latest_detections = []
    low_light_active = False
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop():
    global camera, selected_object, uploaded_video_path, latest_detections, low_light_active
    if camera:
        camera.release()
        camera = None
    if uploaded_video_path and os.path.exists(uploaded_video_path):
        try:
            os.remove(uploaded_video_path)
        except OSError:
            pass
    uploaded_video_path = None
    selected_object = None
    latest_detections = []
    low_light_active = False
    return jsonify({'status': 'stopped'})

@app.route('/select', methods=['POST'])
def select():
    global selected_object, latest_detections
    data = request.json
    x, y = data['x'], data['y']

    best = None
    smallest_area = None
    for det in latest_detections:
        x1, y1, x2, y2 = det['box']
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if smallest_area is None or area < smallest_area:
                smallest_area = area
                best = det

    if best is not None:
        selected_object = {
            'box': best['box'],
            'cls': best['cls'],
            'mask': best.get('mask'),
            'velocity': [0, 0, 0, 0],
            'lost_frames': 0
        }
        return jsonify({'status': 'selected'})

    return jsonify({'status': 'not_found'})

@app.route('/upload', methods=['POST'])
def upload_video():
    global camera, uploaded_video_path, selected_object, latest_detections, low_light_active

    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_ext = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if ext not in allowed_ext:
        return jsonify({'status': 'error', 'message': 'Unsupported video format'}), 400

    if camera:
        camera.release()
        camera = None

    if uploaded_video_path and os.path.exists(uploaded_video_path):
        try:
            os.remove(uploaded_video_path)
        except OSError:
            pass

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    file.save(tmp_file.name)
    tmp_file.close()

    cap = cv2.VideoCapture(tmp_file.name)
    if not cap.isOpened():
        os.remove(tmp_file.name)
        return jsonify({'status': 'error', 'message': 'Failed to open uploaded video'}), 400

    camera = cap
    uploaded_video_path = tmp_file.name
    selected_object = None
    latest_detections = []
    low_light_active = False
    return jsonify({'status': 'uploaded'})

@app.route('/blur', methods=['POST'])
def set_blur():
    global blur_intensity
    blur_intensity = int(request.json['intensity'])
    if blur_intensity < 5:
        blur_intensity = 5
    if blur_intensity > 99:
        blur_intensity = 99
    if blur_intensity % 2 == 0:
        blur_intensity += 1
    return jsonify({'status': 'ok'})

@app.route('/reset', methods=['POST'])
def reset():
    global selected_object
    selected_object = None
    return jsonify({'status': 'reset'})

@app.route('/status', methods=['GET'])
def status():
    global selected_object, latest_detections, low_light_active
    return jsonify({
        'tracking': selected_object is not None,
        'lost_frames': 0 if selected_object is None else int(selected_object.get('lost_frames', 0)),
        'detections': len(latest_detections),
        'low_light_active': bool(low_light_active),
        'occlusion_buffer': MAX_LOST_FRAMES
    })

if __name__ == '__main__':
    print("🚀 Smart Focus AI - http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
