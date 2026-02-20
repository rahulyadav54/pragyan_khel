from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('yolo11n-seg.pt')
selected_box = None
blur_intensity = 25
camera = None

def create_mask(frame, box):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = max(10, (x2 - x1) // 2)
    height = max(10, (y2 - y1) // 2)
    cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    return mask

def apply_blur(frame, mask, intensity):
    blur_size = intensity if intensity % 2 == 1 else intensity + 1
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    mask_norm = mask.astype(float) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=2)
    return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

def generate_frames():
    global selected_box, blur_intensity, camera
    
    while True:
        if camera is None:
            break
        success, frame = camera.read()
        if not success:
            break
        
        results = model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        if selected_box:
            for det in detections:
                if abs(det[0] - selected_box[0]) < 50:
                    selected_box = det
                    mask = create_mask(frame, selected_box)
                    frame = apply_blur(frame, mask, blur_intensity)
                    x1, y1, x2, y2 = selected_box
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 4)
                    break
        
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
    global camera
    camera = cv2.VideoCapture(0)
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop():
    global camera, selected_box
    if camera:
        camera.release()
        camera = None
    selected_box = None
    return jsonify({'status': 'stopped'})

@app.route('/select', methods=['POST'])
def select():
    global selected_box, camera
    data = request.json
    x, y = data['x'], data['y']
    
    if camera:
        success, frame = camera.read()
        if success:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_box = [x1, y1, x2, y2]
                    return jsonify({'status': 'selected'})
    return jsonify({'status': 'not_found'})

@app.route('/blur', methods=['POST'])
def set_blur():
    global blur_intensity
    blur_intensity = int(request.json['intensity'])
    return jsonify({'status': 'ok'})

@app.route('/reset', methods=['POST'])
def reset():
    global selected_box
    selected_box = None
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    print("🚀 Smart Focus AI - http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
