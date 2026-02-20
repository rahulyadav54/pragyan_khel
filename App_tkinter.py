import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL
from PIL import Image, ImageTk
import threading
import os
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available, using fallback detector")

class SimpleTracker:
    """Simple object tracker using IoU"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 30
        
    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {'box': det, 'age': 0}
                self.next_id += 1
            return self.tracks
        
        # Simple IoU matching
        matched = set()
        for det in detections:
            best_iou = 0
            best_id = None
            for tid, track in self.tracks.items():
                iou = self.compute_iou(det, track['box'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = tid
            
            if best_id:
                self.tracks[best_id] = {'box': det, 'age': 0}
                matched.add(best_id)
            else:
                self.tracks[self.next_id] = {'box': det, 'age': 0}
                self.next_id += 1
        
        # Age unmatched tracks
        to_delete = []
        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]
        
        return self.tracks
    
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0

class SimpleDetector:
    def __init__(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2()
        self.names = {0: 'object'}
        
    def __call__(self, frame):
        class DetectionResult:
            def __init__(self):
                self.boxes = []
        
        result = DetectionResult()
        fg_mask = self.back_sub.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                class SimpleBox:
                    def __init__(self, xyxy, conf, cls):
                        self.xyxy = [xyxy]
                        self.conf = [conf]
                        self.cls = [cls]
                box = SimpleBox([x, y, x+w, y+h], 0.5, 0)
                result.boxes.append(box)
        return [result]

class SmartFocusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ZCAM - Smart Focus with AI Blur")
        self.root.geometry("1280x900")
        
        self.capture = None
        self.current_frame = None
        self.processing = False
        self.selected_track_id = None
        self.tracking_active = False
        self.video_path = None
        self.video_thread = None
        self.blur_intensity = 25
        
        self.tracker = SimpleTracker()
        self.current_detections = []
        
        self.init_detector()
        self.build_ui()
        self.update_frame()
        
    def init_detector(self):
        if YOLO_AVAILABLE:
            try:
                self.detector = YOLO('yolo11n-seg.pt')  # Segmentation model
                print("YOLO segmentation model loaded!")
                self.has_segmentation = True
            except:
                try:
                    self.detector = YOLO('yolo11n.pt')
                    print("YOLO detection model loaded!")
                    self.has_segmentation = False
                except:
                    self.detector = SimpleDetector()
                    self.has_segmentation = False
        else:
            self.detector = SimpleDetector()
            self.has_segmentation = False
    
    def build_ui(self):
        # Video canvas
        self.canvas = tk.Canvas(self.root, bg='black', width=1280, height=720)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Blur intensity slider
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(pady=5)
        
        tk.Label(slider_frame, text="Blur Intensity:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.blur_slider = Scale(slider_frame, from_=5, to=51, orient=HORIZONTAL, 
                                 length=300, command=self.update_blur_intensity)
        self.blur_slider.set(25)
        self.blur_slider.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready - Select a video", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.btn_select = tk.Button(btn_frame, text="Select Video", command=self.select_video, 
                                     bg="#4D94CC", fg="white", width=15, height=2)
        self.btn_select.grid(row=0, column=0, padx=5)
        
        self.btn_start = tk.Button(btn_frame, text="Start Processing", command=self.start_processing,
                                    bg="gray", fg="white", width=15, height=2, state=tk.DISABLED)
        self.btn_start.grid(row=0, column=1, padx=5)
        
        self.btn_stop = tk.Button(btn_frame, text="Stop", command=self.stop_processing,
                                   bg="gray", fg="white", width=15, height=2, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=2, padx=5)
    
    def update_blur_intensity(self, val):
        self.blur_intensity = int(val)
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1
    
    def select_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path = filepath
            self.btn_start.config(state=tk.NORMAL, bg="#33B533")
            self.status_label.config(text=f"Selected: {os.path.basename(filepath)}")
            
            if self.capture:
                self.capture.release()
            self.capture = cv2.VideoCapture(filepath)
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
    
    def start_processing(self):
        if not self.capture or not self.capture.isOpened():
            return
        
        self.processing = True
        self.tracking_active = False
        self.selected_track_id = None
        self.tracker = SimpleTracker()
        
        self.btn_start.config(state=tk.DISABLED, bg="gray")
        self.btn_stop.config(state=tk.NORMAL, bg="#CC3333")
        self.btn_select.config(state=tk.DISABLED, bg="gray")
        self.status_label.config(text="Processing - Click on an object to track and blur background")
        
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
    
    def stop_processing(self):
        self.processing = False
        self.tracking_active = False
        
        self.btn_start.config(state=tk.NORMAL, bg="#33B533")
        self.btn_stop.config(state=tk.DISABLED, bg="gray")
        self.btn_select.config(state=tk.NORMAL, bg="#4D94CC")
        self.status_label.config(text="Stopped - Select a video")
    
    def on_canvas_click(self, event):
        if not self.processing or self.current_frame is None:
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if not hasattr(self, 'last_display_size'):
            return
        
        img_h, img_w = self.last_display_size
        
        # Calculate offset for centered image
        offset_x = (canvas_w - img_w) // 2
        offset_y = (canvas_h - img_h) // 2
        
        # Adjust click coordinates
        click_x = event.x - offset_x
        click_y = event.y - offset_y
        
        if click_x < 0 or click_y < 0 or click_x >= img_w or click_y >= img_h:
            return
        
        # Scale to original frame size
        orig_h, orig_w = self.current_frame.shape[:2]
        scale_x = orig_w / img_w
        scale_y = orig_h / img_h
        
        img_x = int(click_x * scale_x)
        img_y = int(click_y * scale_y)
        
        self.select_object_at_position(img_x, img_y)
    
    def select_object_at_position(self, x, y):
        if not hasattr(self, 'current_tracks'):
            return
        
        for tid, track in self.current_tracks.items():
            box = track['box']
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_track_id = tid
                self.tracking_active = True
                self.status_label.config(text=f"Tracking object ID: {tid} - Click another to switch")
                return
    
    def create_mask_from_box(self, frame, box):
        """Create segmentation mask from bounding box"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = map(int, box)
        
        # Create elliptical mask for more natural look
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = (x2 - x1) // 2
        height = (y2 - y1) // 2
        
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        return mask
    
    def apply_smart_blur(self, frame, mask):
        """Apply cinematic blur to background"""
        # Create blurred version
        blurred = cv2.GaussianBlur(frame, (self.blur_intensity, self.blur_intensity), 0)
        
        # Normalize mask
        mask_norm = mask.astype(float) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        # Blend sharp subject with blurred background
        result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def process_video(self):
        while self.processing:
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                try:
                    # Run detection
                    results = self.detector(frame, verbose=False)[0]
                    
                    # Extract detections
                    detections = []
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append([x1, y1, x2, y2])
                    
                    # Update tracker
                    self.current_tracks = self.tracker.update(detections)
                    
                    # Apply blur if tracking
                    if self.tracking_active and self.selected_track_id in self.current_tracks:
                        tracked_box = self.current_tracks[self.selected_track_id]['box']
                        
                        # Create mask
                        if self.has_segmentation and hasattr(results, 'masks') and results.masks is not None:
                            # Use actual segmentation mask if available
                            for i, box in enumerate(results.boxes):
                                box_coords = list(map(int, box.xyxy[0]))
                                if box_coords == tracked_box:
                                    mask = results.masks.data[i].cpu().numpy()
                                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                                    mask = (mask * 255).astype(np.uint8)
                                    break
                            else:
                                mask = self.create_mask_from_box(frame, tracked_box)
                        else:
                            mask = self.create_mask_from_box(frame, tracked_box)
                        
                        # Apply blur
                        frame = self.apply_smart_blur(frame, mask)
                    
                    self.current_frame = frame
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    self.current_frame = frame
                    time.sleep(0.033)
    
    def display_frame(self, frame):
        if frame is None:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        scale = min(canvas_w/w, canvas_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        self.last_display_size = (new_h, new_w)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(image=img)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.photo)
    
    def update_frame(self):
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
        self.root.after(33, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartFocusApp(root)
    root.mainloop()
