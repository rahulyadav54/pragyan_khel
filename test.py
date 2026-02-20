import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
import threading
from queue import Queue
import os
import time
import sys
from collections import OrderedDict  # Fix for OrderedDict import

# Fix for torchvision import issues
import warnings
warnings.filterwarnings("ignore")

# Alternative YOLO import with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO import failed: {e}")
    print("Using fallback detection mode")
    YOLO_AVAILABLE = False

# Check Python version
print(f"Python version: {sys.version}")

class SimpleDetector:
    """Fallback detector when YOLO is not available"""
    def __init__(self):
        self.names = {0: 'person', 1: 'object'}
        # Initialize background subtractor for motion detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2()
        
    def __call__(self, frame):
        # Create a simple detection result structure
        class DetectionResult:
            def __init__(self):
                self.boxes = []
                
        result = DetectionResult()
        
        # Use background subtraction to detect moving objects
        fg_mask = self.back_sub.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        class Box:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = xyxy
                self.conf = conf
                self.cls = cls
                
        result.boxes = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                # Create a box object with required attributes
                box = Box(
                    xyxy=[[[x, y, x+w, y+h]]],  # Format expected by the code
                    conf=[0.5],
                    cls=[0]
                )
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
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.video_path = None
        
        # Initialize models
        self.init_models()
        
        # Tracking variables
        self.tracks = {}
        self.next_track_id = 0
        self.active_track_id = None
        self.reid_features = {}
        
    def init_models(self):
        """Initialize all AI models"""
        print("Loading detection model...")
        
        if YOLO_AVAILABLE:
            try:
                # Try to load YOLO model
                self.detector = YOLO('yolo11n.pt')
                print("YOLO model loaded successfully!")
            except Exception as e:
                print(f"YOLO loading failed: {e}")
                print("Using fallback detector")
                self.detector = SimpleDetector()
        else:
            print("Using fallback detector")
            self.detector = SimpleDetector()
        
        # Simple tracker implementation
        self.tracker = SimpleTracker()
        
        print("Models initialized successfully!")
        
    def build(self):
        """Build the UI"""
        # Main layout
        self.layout = BoxLayout(orientation='vertical')
        
        # Video display widget
        self.video_widget = Image(size_hint=(1, 0.9))
        self.layout.add_widget(self.video_widget)
        
        # Status label
        self.status_label = Label(size_hint=(1, 0.05), text="Ready - Select a video")
        self.layout.add_widget(self.status_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        
        self.btn_select = Button(text='Select Video')
        self.btn_select.bind(on_press=self.select_video)
        button_layout.add_widget(self.btn_select)
        
        self.btn_start = Button(text='Start Processing', disabled=True)
        self.btn_start.bind(on_press=self.start_processing)
        button_layout.add_widget(self.btn_start)
        
        self.btn_stop = Button(text='Stop', disabled=True)
        self.btn_stop.bind(on_press=self.stop_processing)
        button_layout.add_widget(self.btn_stop)
        
        self.layout.add_widget(button_layout)
        
        # Bind touch events for object selection
        self.video_widget.bind(on_touch_down=self.on_video_touch)
        
        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0/30.0)  # 30 FPS
        
        return self.layout
    
    def select_video(self, instance):
        """Open file chooser to select video"""
        content = BoxLayout(orientation='vertical')
        
        # Simple file dialog using Kivy's FileChooser
        filechooser = FileChooserListView(path=os.path.expanduser("~"))
        filechooser.filters = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        content.add_widget(filechooser)
        
        btn_layout = BoxLayout(size_hint_y=0.2, spacing=10)
        btn_select = Button(text='Select')
        btn_cancel = Button(text='Cancel')
        btn_layout.add_widget(btn_select)
        btn_layout.add_widget(btn_cancel)
        content.add_widget(btn_layout)
        
        popup = Popup(title='Select Video File', content=content, 
                     size_hint=(0.9, 0.9))
        
        def on_select(instance):
            if filechooser.selection and filechooser.selection[0]:
                self.video_path = filechooser.selection[0]
                self.btn_start.disabled = False
                self.status_label.text = f"Selected: {os.path.basename(self.video_path)}"
                popup.dismiss()
                
                # Open video to get first frame
                if self.capture:
                    self.capture.release()
                self.capture = cv2.VideoCapture(self.video_path)
                ret, frame = self.capture.read()
                if ret:
                    self.current_frame = frame
                    self.display_frame(frame)
                    
        btn_select.bind(on_press=on_select)
        btn_cancel.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def start_processing(self, instance):
        """Start video processing"""
        self.processing = True
        self.btn_start.disabled = True
        self.btn_stop.disabled = False
        self.btn_select.disabled = True
        self.status_label.text = "Processing - Tap on an object to track"
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self, instance):
        """Stop video processing"""
        self.processing = False
        self.btn_start.disabled = False
        self.btn_stop.disabled = True
        self.btn_select.disabled = False
        self.tracking_active = False
        self.selected_object = None
        self.status_label.text = "Stopped - Select a video"
        
    def on_video_touch(self, instance, touch):
        """Handle touch events on video for object selection"""
        if self.processing and self.current_frame is not None and touch.is_mouse_scrolling == False:
            # Check if touch is within video widget bounds
            if touch.x <= self.video_widget.width and touch.y <= self.video_widget.height:
                # Convert touch coordinates to image coordinates
                img_x = int((touch.x / self.video_widget.width) * self.current_frame.shape[1])
                img_y = int(((self.video_widget.height - touch.y) / self.video_widget.height) * self.current_frame.shape[0])
                
                # Select object at touched position
                self.select_object_at_position(img_x, img_y)
                return True
        return False
    
    def select_object_at_position(self, x, y):
        """Select object at given position"""
        if self.current_frame is None:
            return
            
        # Run detection on current frame
        try:
            results = self.detector(self.current_frame)[0]
            
            # Find object containing the clicked point
            object_found = False
            for box in results.boxes:
                # Handle different box formats
                if hasattr(box, 'xyxy'):
                    box_data = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                else:
                    continue
                    
                x1, y1, x2, y2 = box_data.astype(int)
                
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Object found
                    class_id = int(box.cls[0]) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                    confidence = float(box.conf[0]) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                    class_name = results.names[class_id] if hasattr(results, 'names') else f"Object_{class_id}"
                    
                    # Extract ROI
                    roi = self.current_frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        self.selected_object = {
                            'bbox': [x1, y1, x2, y2],
                            'class': class_name,
                            'confidence': confidence,
                            'features': self.extract_features(roi)
                        }
                        
                        # Assign new track ID
                        self.active_track_id = self.next_track_id
                        self.next_track_id += 1
                        
                        # Store features for ReID
                        self.reid_features[self.active_track_id] = self.selected_object['features']
                        
                        self.tracking_active = True
                        self.status_label.text = f"Tracking: {class_name} (ID: {self.active_track_id})"
                        print(f"Selected {class_name} with confidence {confidence:.2f}")
                        object_found = True
                        break
            
            if not object_found:
                self.status_label.text = "No object detected at tap location"
                
        except Exception as e:
            print(f"Error in object selection: {e}")
            self.status_label.text = "Error selecting object"
    
    def extract_features(self, roi):
        """Extract features from ROI for re-identification"""
        try:
            # Simple feature extraction using color histograms
            if roi.size == 0:
                return None
                
            # Convert to HSV for better color representation
            if len(roi.shape) == 3:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # Calculate color histogram
                hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                return hist.flatten()
            return None
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def match_features(self, features1, features2):
        """Match features for re-identification"""
        if features1 is None or features2 is None:
            return 0
        try:
            # Use correlation as similarity measure
            return cv2.compareHist(features1.astype(np.float32), 
                                  features2.astype(np.float32), 
                                  cv2.HISTCMP_CORREL)
        except:
            return 0
    
    def process_frames(self):
        """Main processing loop running in separate thread"""
        frame_count = 0
        while self.processing:
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    # Loop video
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process every other frame for better performance
                frame_count += 1
                if frame_count % 2 == 0:
                    # Process frame
                    processed_frame = self.process_single_frame(frame)
                else:
                    processed_frame = frame
                
                # Store for display
                if not self.result_queue.full():
                    self.result_queue.put(processed_frame)
    
    def process_single_frame(self, frame):
        """Process a single frame"""
        processed_frame = frame.copy()
        
        if self.tracking_active and self.selected_object:
            try:
                # Run detection
                results = self.detector(frame)[0]
                
                # Find best match for selected object
                best_match = None
                best_score = 0
                
                for box in results.boxes:
                    # Handle different box formats
                    if hasattr(box, 'xyxy'):
                        box_data = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                    else:
                        continue
                        
                    x1, y1, x2, y2 = box_data.astype(int)
                    
                    # Extract features from detected object
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        features = self.extract_features(roi)
                        
                        # Match with selected object
                        if features is not None and self.selected_object['features'] is not None:
                            score = self.match_features(features, self.selected_object['features'])
                            
                            # Simple IoU with previous position for continuity
                            prev_bbox = self.selected_object['bbox']
                            iou = self.calculate_iou(prev_bbox, [x1, y1, x2, y2])
                            
                            # Combine scores
                            combined_score = score * 0.7 + iou * 0.3
                            
                            if combined_score > 0.3 and combined_score > best_score:
                                best_score = combined_score
                                best_match = [x1, y1, x2, y2]
                                best_class = results.names[int(box.cls[0])] if hasattr(results, 'names') else "Object"
                
                if best_match:
                    # Update selected object position
                    self.selected_object['bbox'] = best_match
                    self.selected_object['class'] = best_class
                    
                    # Apply effects
                    processed_frame = self.apply_background_blur(frame, best_match)
                    processed_frame = self.highlight_object(processed_frame, best_match)
                else:
                    # Object lost, show warning
                    cv2.putText(processed_frame, "Object Lost - Searching...", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                processed_frame = frame
        
        return processed_frame
    
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
    
    def apply_background_blur(self, frame, object_bbox):
        """Apply Gaussian blur to everything except the selected object"""
        if frame is None:
            return frame
            
        try:
            # Create mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = object_bbox
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            mask[y1:y2, x1:x2] = 255
            
            # Apply heavy blur to entire frame
            blurred = cv2.GaussianBlur(frame, (51, 51), 30)
            
            # Combine blurred background with sharp foreground
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (frame * mask_3channel + blurred * (1 - mask_3channel)).astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"Blur error: {e}")
            return frame
    
    def highlight_object(self, frame, bbox):
        """Highlight the tracked object with a bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label
            if self.selected_object:
                label = f"{self.selected_object.get('class', 'Object')} (ID: {self.active_track_id})"
                # Add background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1+w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add focus indicator
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, (0, 255, 0), -1)
            
            return frame
        except Exception as e:
            print(f"Highlight error: {e}")
            return frame
    
    def display_frame(self, frame):
        """Display frame in Kivy widget"""
        if frame is None:
            return
            
        try:
            # Resize frame if too large
            height, width = frame.shape[:2]
            max_size = 800
            if width > max_size:
                scale = max_size / width
                new_width = max_size
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert frame to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.video_widget.texture = texture
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_frame(self, dt):
        """Update video display"""
        try:
            if not self.result_queue.empty():
                frame = self.result_queue.get()
                self.display_frame(frame)
                self.current_frame = frame
            elif self.capture and not self.processing and self.capture.isOpened():
                # Display current frame when not processing
                ret, frame = self.capture.read()
                if ret:
                    self.display_frame(frame)
                    self.current_frame = frame
                else:
                    # Reset video to beginning
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as e:
            print(f"Update error: {e}")
    
    def on_stop(self):
        """Cleanup when app closes"""
        self.processing = False
        if self.capture:
            self.capture.release()


class SimpleTracker:
    """Simple object tracker implementation"""
    def __init__(self, max_lost=10):
        self.tracks = []
        self.next_id = 0
        self.max_lost = max_lost
        
    def update(self, detections):
        """Update tracks with new detections"""
        # Update existing tracks
        for track in self.tracks:
            if track.get('is_active', False):
                track['lost_frames'] = track.get('lost_frames', 0) + 1
        
        # Match detections to tracks
        for detection in detections:
            best_match = None
            best_iou = 0
            
            for track in self.tracks:
                if track.get('is_active', False):
                    iou = self.calculate_iou(track['bbox'], detection['bbox'])
                    if iou > 0.3 and iou > best_iou:
                        best_iou = iou
                        best_match = track
            
            if best_match:
                # Update existing track
                best_match['bbox'] = detection['bbox']
                best_match['lost_frames'] = 0
                best_match['confidence'] = detection['confidence']
            else:
                # Create new track
                new_track = {
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class': detection.get('class', 0),
                    'lost_frames': 0,
                    'is_active': True
                }
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Mark lost tracks as inactive
        for track in self.tracks:
            if track.get('lost_frames', 0) > self.max_lost:
                track['is_active'] = False
    
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


if __name__ == '__main__':
    SmartFocusApp().run()