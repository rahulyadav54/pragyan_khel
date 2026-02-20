import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
import threading
from queue import Queue
import os
import time

# Try to import YOLO, but don't fail if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available, using fallback detector")

class SimpleDetector:
    """Fallback detector using background subtraction"""
    def __init__(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2()
        self.names = {0: 'object'}
        
    def __call__(self, frame):
        class DetectionResult:
            def __init__(self):
                self.boxes = []
                self.names = {0: 'object'}
        
        result = DetectionResult()
        result.boxes = []
        
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create a simple box object
                class SimpleBox:
                    def __init__(self, xyxy, conf, cls):
                        self.xyxy = [xyxy]
                        self.conf = [conf]
                        self.cls = [cls]
                
                box = SimpleBox([x, y, x+w, y+h], 0.5, 0)
                result.boxes.append(box)
        
        return [result]

class SmartFocusApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Video processing variables
        self.capture = None
        self.current_frame = None
        self.processing = False
        self.selected_object = None
        self.tracking_active = False
        self.result_queue = Queue(maxsize=2)
        self.video_path = None
        self.video_thread = None
        
        # Initialize detector
        self.init_detector()
        
        # Tracking variables
        self.tracked_object = None
        self.track_id = 0
        
    def init_detector(self):
        """Initialize the detector"""
        if YOLO_AVAILABLE:
            try:
                self.detector = YOLO('yolo11n.pt')
                print("YOLO model loaded successfully!")
            except:
                print("Failed to load YOLO, using fallback detector")
                self.detector = SimpleDetector()
        else:
            self.detector = SimpleDetector()
            print("Using fallback detector")
    
    def build(self):
        """Build the UI"""
        # Main layout
        self.layout = BoxLayout(orientation='vertical')
        
        # Video display widget
        self.video_widget = Image(size_hint=(1, 0.85))
        self.layout.add_widget(self.video_widget)
        
        # Status label
        self.status_label = Label(
            size_hint=(1, 0.05),
            text="Ready - Select a video",
            color=(1, 1, 1, 1)
        )
        self.layout.add_widget(self.status_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        
        self.btn_select = Button(text='Select Video', background_color=(0.3, 0.6, 0.9, 1))
        self.btn_select.bind(on_press=self.select_video)
        button_layout.add_widget(self.btn_select)
        
        self.btn_start = Button(
            text='Start Processing', 
            disabled=True,
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.btn_start.bind(on_press=self.start_processing)
        button_layout.add_widget(self.btn_start)
        
        self.btn_stop = Button(
            text='Stop', 
            disabled=True,
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.btn_stop.bind(on_press=self.stop_processing)
        button_layout.add_widget(self.btn_stop)
        
        self.layout.add_widget(button_layout)
        
        # Bind touch events
        self.video_widget.bind(on_touch_down=self.on_video_touch)
        
        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0/30.0)
        
        return self.layout
    
    def select_video(self, instance):
        """Open file chooser to select video"""
        content = BoxLayout(orientation='vertical')
        
        # Create file chooser
        filechooser = FileChooserListView(
            path=os.path.expanduser("~"),
            filters=['*.mp4', '*.avi', '*.mov', '*.mkv']
        )
        content.add_widget(filechooser)
        
        # Button layout
        btn_layout = BoxLayout(size_hint_y=0.2, spacing=10, padding=10)
        btn_select = Button(text='Select', background_color=(0.2, 0.7, 0.2, 1))
        btn_cancel = Button(text='Cancel', background_color=(0.7, 0.2, 0.2, 1))
        btn_layout.add_widget(btn_select)
        btn_layout.add_widget(btn_cancel)
        content.add_widget(btn_layout)
        
        # Create popup
        popup = Popup(
            title='Select Video File',
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        def on_select(instance):
            if filechooser.selection and filechooser.selection[0]:
                self.video_path = filechooser.selection[0]
                self.btn_start.disabled = False
                self.btn_start.background_color = (0.2, 0.7, 0.2, 1)
                self.status_label.text = f"Selected: {os.path.basename(self.video_path)}"
                popup.dismiss()
                
                # Open video
                if self.capture:
                    self.capture.release()
                self.capture = cv2.VideoCapture(self.video_path)
                
                # Get first frame
                ret, frame = self.capture.read()
                if ret:
                    self.current_frame = frame
                    self.display_frame(frame)
        
        btn_select.bind(on_press=on_select)
        btn_cancel.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def start_processing(self, instance):
        """Start video processing"""
        if not self.capture or not self.capture.isOpened():
            return
            
        self.processing = True
        self.tracking_active = False
        self.selected_object = None
        
        # Update UI
        self.btn_start.disabled = True
        self.btn_start.background_color = (0.5, 0.5, 0.5, 1)
        self.btn_stop.disabled = False
        self.btn_stop.background_color = (0.7, 0.2, 0.2, 1)
        self.btn_select.disabled = True
        self.btn_select.background_color = (0.5, 0.5, 0.5, 1)
        
        self.status_label.text = "Processing - Tap on an object to track"
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_processing(self, instance):
        """Stop video processing"""
        self.processing = False
        self.tracking_active = False
        
        # Update UI
        self.btn_start.disabled = False
        self.btn_start.background_color = (0.2, 0.7, 0.2, 1)
        self.btn_stop.disabled = True
        self.btn_stop.background_color = (0.5, 0.5, 0.5, 1)
        self.btn_select.disabled = False
        self.btn_select.background_color = (0.3, 0.6, 0.9, 1)
        
        self.status_label.text = "Stopped - Select a video"
    
    def on_video_touch(self, instance, touch):
        """Handle touch events for object selection"""
        if not self.processing or self.current_frame is None:
            return False
            
        # Check if touch is within video widget
        if touch.x <= self.video_widget.width and touch.y <= self.video_widget.height:
            # Convert coordinates
            img_x = int((touch.x / self.video_widget.width) * self.current_frame.shape[1])
            img_y = int(((self.video_widget.height - touch.y) / self.video_widget.height) * self.current_frame.shape[0])
            
            # Select object
            self.select_object_at_position(img_x, img_y)
            return True
        return False
    
    def select_object_at_position(self, x, y):
        """Select object at clicked position"""
        if self.current_frame is None:
            return
            
        try:
            # Run detection
            results = self.detector(self.current_frame)[0]
            
            # Find object at click position
            for box in results.boxes:
                # Get box coordinates
                if hasattr(box, 'xyxy'):
                    xyxy = box.xyxy[0]
                    if hasattr(xyxy, 'cpu'):
                        xyxy = xyxy.cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                else:
                    continue
                
                # Check if click is inside box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Get object info
                    class_id = int(box.cls[0]) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                    confidence = float(box.conf[0]) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                    
                    # Get class name
                    if hasattr(results, 'names'):
                        class_name = results.names.get(class_id, f'Object_{class_id}')
                    else:
                        class_name = f'Object_{class_id}'
                    
                    # Extract ROI for features
                    roi = self.current_frame[y1:y2, x1:x2]
                    features = self.extract_features(roi)
                    
                    # Create selected object
                    self.selected_object = {
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': confidence,
                        'features': features,
                        'track_id': self.track_id
                    }
                    
                    self.track_id += 1
                    self.tracking_active = True
                    
                    self.status_label.text = f"Tracking: {class_name}"
                    print(f"Selected {class_name} at ({x}, {y})")
                    break
            
        except Exception as e:
            print(f"Selection error: {e}")
    
    def extract_features(self, roi):
        """Extract color histogram features"""
        if roi.size == 0:
            return None
            
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Calculate histogram
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except:
            return None
    
    def match_features(self, features1, features2):
        """Match features using histogram correlation"""
        if features1 is None or features2 is None:
            return 0
        try:
            return cv2.compareHist(
                features1.astype(np.float32),
                features2.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
        except:
            return 0
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        except:
            return 0
    
    def process_video(self):
        """Process video frames in thread"""
        while self.processing and self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                # Loop video
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process frame
            processed = self.process_frame(frame)
            
            # Add to queue
            if not self.result_queue.full():
                self.result_queue.put(processed)
    
    def process_frame(self, frame):
        """Process a single frame"""
        result = frame.copy()
        
        if self.tracking_active and self.selected_object:
            try:
                # Run detection
                detections = self.detector(frame)[0]
                
                # Find best match for selected object
                best_match = None
                best_score = 0
                
                for box in detections.boxes:
                    # Get box coordinates
                    if hasattr(box, 'xyxy'):
                        xyxy = box.xyxy[0]
                        if hasattr(xyxy, 'cpu'):
                            xyxy = xyxy.cpu().numpy()
                        x1, y1, x2, y2 = xyxy.astype(int)
                    else:
                        continue
                    
                    # Extract features
                    roi = frame[y1:y2, x1:x2]
                    features = self.extract_features(roi)
                    
                    if features is not None:
                        # Calculate scores
                        feat_score = self.match_features(features, self.selected_object['features'])
                        iou_score = self.calculate_iou(self.selected_object['bbox'], [x1, y1, x2, y2])
                        
                        # Combined score
                        combined = feat_score * 0.6 + iou_score * 0.4
                        
                        if combined > 0.3 and combined > best_score:
                            best_score = combined
                            best_match = [x1, y1, x2, y2]
                
                if best_match:
                    # Update object position
                    self.selected_object['bbox'] = best_match
                    
                    # Apply effects
                    result = self.apply_background_blur(frame, best_match)
                    result = self.highlight_object(result, best_match)
                else:
                    # Object lost
                    cv2.putText(result, "Object Lost", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
        
        return result
    
    def apply_background_blur(self, frame, bbox):
        """Apply blur to background"""
        try:
            # Create mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are valid
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
                
                # Apply blur
                blurred = cv2.GaussianBlur(frame, (31, 31), 20)
                
                # Combine
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                result = (frame * mask_3channel + blurred * (1 - mask_3channel)).astype(np.uint8)
                return result
            
        except Exception as e:
            print(f"Blur error: {e}")
        
        return frame
    
    def highlight_object(self, frame, bbox):
        """Highlight tracked object"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are valid
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label
            if self.selected_object:
                label = f"{self.selected_object['class']}"
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Highlight error: {e}")
        
        return frame
    
    def display_frame(self, frame):
        """Display frame in Kivy widget"""
        if frame is None:
            return
            
        try:
            # Resize if too large
            height, width = frame.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            # Update widget
            self.video_widget.texture = texture
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_frame(self, dt):
        """Update frame from queue"""
        if not self.result_queue.empty():
            frame = self.result_queue.get()
            self.display_frame(frame)
            self.current_frame = frame
        elif self.capture and not self.processing and self.capture.isOpened():
            # Show static frames when not processing
            ret, frame = self.capture.read()
            if ret:
                self.display_frame(frame)
                self.current_frame = frame
            else:
                # Reset video
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def on_stop(self):
        """Cleanup on exit"""
        self.processing = False
        if self.capture:
            self.capture.release()

if __name__ == '__main__':
    SmartFocusApp().run()