import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
import threading
from queue import Queue
import os

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
        
        # Initialize detector (simple motion-based detector)
        self.detector = SimpleDetector()
        
        # Tracking variables
        self.tracked_object = None
        self.track_id = 0
        
    def build(self):
        """Build the UI"""
        # Set window size
        Window.size = (1024, 768)
        
        # Main layout
        self.layout = BoxLayout(orientation='vertical', spacing=5, padding=5)
        
        # Video display widget with border
        self.video_widget = Image(
            size_hint=(1, 0.8),
            allow_stretch=True,
            keep_ratio=True
        )
        self.layout.add_widget(self.video_widget)
        
        # Status label
        self.status_label = Label(
            size_hint=(1, 0.05),
            text="Ready - Click 'Select Video' to begin",
            color=(0, 1, 0, 1),
            font_size='16sp'
        )
        self.layout.add_widget(self.status_label)
        
        # Instructions label
        self.instructions_label = Label(
            size_hint=(1, 0.03),
            text="",
            color=(1, 1, 0, 1),
            font_size='14sp'
        )
        self.layout.add_widget(self.instructions_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint=(1, 0.12), spacing=10)
        
        self.btn_select = Button(
            text='📁 Select Video',
            font_size='18sp',
            background_color=(0.2, 0.6, 1, 1),
            size_hint_x=0.33
        )
        self.btn_select.bind(on_press=self.select_video)
        button_layout.add_widget(self.btn_select)
        
        self.btn_start = Button(
            text='▶ Start Processing',
            font_size='18sp',
            background_color=(0.5, 0.5, 0.5, 1),
            disabled=True,
            size_hint_x=0.33
        )
        self.btn_start.bind(on_press=self.start_processing)
        button_layout.add_widget(self.btn_start)
        
        self.btn_stop = Button(
            text='⏹ Stop',
            font_size='18sp',
            background_color=(0.5, 0.5, 0.5, 1),
            disabled=True,
            size_hint_x=0.33
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
        # Create file chooser content
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Add label
        content.add_widget(Label(
            text='Select a video file (MP4, AVI, MOV, MKV)',
            size_hint_y=0.1,
            color=(1, 1, 1, 1)
        ))
        
        # Create file chooser
        filechooser = FileChooserListView(
            path=os.path.expanduser("~"),
            filters=['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV', '*.mkv', '*.MKV'],
            size_hint_y=0.7
        )
        content.add_widget(filechooser)
        
        # Button layout
        btn_layout = BoxLayout(size_hint_y=0.2, spacing=10, padding=10)
        btn_select = Button(
            text='Select',
            background_color=(0, 1, 0, 1),
            font_size='18sp'
        )
        btn_cancel = Button(
            text='Cancel',
            background_color=(1, 0, 0, 1),
            font_size='18sp'
        )
        btn_layout.add_widget(btn_select)
        btn_layout.add_widget(btn_cancel)
        content.add_widget(btn_layout)
        
        # Create popup
        self.popup = Popup(
            title='Select Video File',
            content=content,
            size_hint=(0.9, 0.9),
            auto_dismiss=False
        )
        
        def on_select(instance):
            if filechooser.selection and len(filechooser.selection) > 0:
                self.video_path = filechooser.selection[0]
                self.load_video()
                self.popup.dismiss()
        
        def on_cancel(instance):
            self.popup.dismiss()
        
        btn_select.bind(on_press=on_select)
        btn_cancel.bind(on_press=on_cancel)
        
        self.popup.open()
    
    def load_video(self):
        """Load the selected video"""
        try:
            # Release previous capture if any
            if self.capture:
                self.capture.release()
            
            # Open new video
            self.capture = cv2.VideoCapture(self.video_path)
            
            if self.capture.isOpened():
                # Get first frame
                ret, frame = self.capture.read()
                if ret:
                    self.current_frame = frame
                    self.display_frame(frame)
                    
                    # Enable start button
                    self.btn_start.disabled = False
                    self.btn_start.background_color = (0, 1, 0, 1)
                    
                    # Update status
                    filename = os.path.basename(self.video_path)
                    self.status_label.text = f"✓ Loaded: {filename}"
                    self.status_label.color = (0, 1, 0, 1)
                    self.instructions_label.text = "Click 'Start Processing' to begin"
                    
        except Exception as e:
            self.show_error(f"Error loading video: {str(e)}")
    
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
        self.btn_stop.background_color = (1, 0, 0, 1)
        self.btn_select.disabled = True
        self.btn_select.background_color = (0.5, 0.5, 0.5, 1)
        
        self.status_label.text = "▶ Processing - Tap on an object to track"
        self.status_label.color = (1, 1, 0, 1)
        self.instructions_label.text = "Tap on any object in the video to start tracking"
        
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
        self.btn_start.background_color = (0, 1, 0, 1)
        self.btn_stop.disabled = True
        self.btn_stop.background_color = (0.5, 0.5, 0.5, 1)
        self.btn_select.disabled = False
        self.btn_select.background_color = (0.2, 0.6, 1, 1)
        
        self.status_label.text = "⏸ Processing stopped"
        self.status_label.color = (1, 1, 1, 1)
        self.instructions_label.text = ""
    
    def on_video_touch(self, instance, touch):
        """Handle touch events for object selection"""
        if not self.processing or self.current_frame is None:
            return False
            
        # Check if touch is within video widget
        if instance.collide_point(touch.x, touch.y):
            # Convert coordinates
            img_x = int((touch.x / instance.width) * self.current_frame.shape[1])
            img_y = int(((instance.height - touch.y) / instance.height) * self.current_frame.shape[0])
            
            # Ensure coordinates are within bounds
            img_x = max(0, min(img_x, self.current_frame.shape[1] - 1))
            img_y = max(0, min(img_y, self.current_frame.shape[0] - 1))
            
            # Select object
            self.select_object_at_position(img_x, img_y)
            return True
        return False
    
    def select_object_at_position(self, x, y):
        """Select object at clicked position"""
        if self.current_frame is None:
            return
            
        try:
            # Detect objects in frame
            objects = self.detector.detect(self.current_frame)
            
            # Find object at click position
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Extract ROI for features
                    roi = self.current_frame[y1:y2, x1:x2]
                    
                    # Create selected object
                    self.selected_object = {
                        'bbox': [x1, y1, x2, y2],
                        'class': obj['class'],
                        'features': self.extract_features(roi),
                        'track_id': self.track_id
                    }
                    
                    self.track_id += 1
                    self.tracking_active = True
                    
                    self.status_label.text = f"✓ Tracking: {obj['class']}"
                    self.instructions_label.text = "Object locked - background blur active"
                    print(f"Selected object at ({x}, {y})")
                    break
            else:
                self.status_label.text = "✗ No object detected at tap location"
                
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
        frame_count = 0
        while self.processing and self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                # Loop video
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process every other frame for better performance
            frame_count += 1
            if frame_count % 2 == 0:
                processed = self.process_frame(frame)
            else:
                processed = frame
            
            # Add to queue
            if not self.result_queue.full():
                self.result_queue.put(processed)
    
    def process_frame(self, frame):
        """Process a single frame"""
        result = frame.copy()
        
        if self.tracking_active and self.selected_object:
            try:
                # Detect objects
                objects = self.detector.detect(frame)
                
                # Find best match for selected object
                best_match = None
                best_score = 0
                
                for obj in objects:
                    # Calculate scores
                    x1, y1, x2, y2 = obj['bbox']
                    roi = frame[y1:y2, x1:x2]
                    features = self.extract_features(roi)
                    
                    if features is not None:
                        feat_score = self.match_features(features, self.selected_object['features'])
                        iou_score = self.calculate_iou(self.selected_object['bbox'], obj['bbox'])
                        
                        # Combined score
                        combined = feat_score * 0.6 + iou_score * 0.4
                        
                        if combined > 0.3 and combined > best_score:
                            best_score = combined
                            best_match = obj['bbox']
                
                if best_match:
                    # Update object position
                    self.selected_object['bbox'] = best_match
                    
                    # Apply effects
                    result = self.apply_background_blur(frame, best_match)
                    result = self.highlight_object(result, best_match)
                else:
                    # Object lost
                    cv2.putText(result, "🔍 Searching...", (50, 50),
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
                # Draw filled rectangle on mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Apply blur
                blurred = cv2.GaussianBlur(frame, (51, 51), 30)
                
                # Combine using mask
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
            
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label background
            if self.selected_object:
                label = f"🎯 {self.selected_object['class']}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1-30), (x1+w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1-8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
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
            # Show frames when not processing
            ret, frame = self.capture.read()
            if ret:
                self.display_frame(frame)
                self.current_frame = frame
            else:
                # Reset video
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def show_error(self, message):
        """Show error popup"""
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text=message, color=(1, 0, 0, 1)))
        
        btn = Button(text='OK', size_hint_y=0.3, background_color=(0, 1, 0, 1))
        content.add_widget(btn)
        
        popup = Popup(title='Error', content=content, size_hint=(0.6, 0.4))
        btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def on_stop(self):
        """Cleanup on exit"""
        self.processing = False
        if self.capture:
            self.capture.release()


class SimpleDetector:
    """Simple motion and color-based detector"""
    def __init__(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2()
        self.object_classes = ['person', 'vehicle', 'animal', 'object']
        
    def detect(self, frame):
        """Detect objects in frame"""
        objects = []
        
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Determine object class based on aspect ratio and size
                aspect_ratio = h / w if w > 0 else 0
                if 1.5 < aspect_ratio < 3.0 and area > 5000:
                    obj_class = 'person'
                elif w > h and area > 10000:
                    obj_class = 'vehicle'
                elif area > 2000:
                    obj_class = 'animal'
                else:
                    obj_class = 'object'
                
                objects.append({
                    'bbox': [x, y, x+w, y+h],
                    'class': obj_class,
                    'area': area
                })
        
        return objects


if __name__ == '__main__':
    SmartFocusApp().run()