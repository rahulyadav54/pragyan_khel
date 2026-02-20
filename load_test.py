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
import threading
from queue import Queue
import urllib.request
import os
import time

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
        self.frame_count = 0
        
        # Initialize detector
        self.detector = SimpleDetector()
        
    def build(self):
        """Build the UI"""
        # Set window size
        Window.size = (1000, 700)
        Window.clearcolor = (0.2, 0.2, 0.2, 1)
        
        # Main layout
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Title
        title = Label(
            text="SmartFocus AI Tracker",
            size_hint=(1, 0.08),
            font_size='24sp',
            color=(0.2, 0.8, 1, 1),
            bold=True
        )
        self.layout.add_widget(title)
        
        # Video display widget
        self.video_widget = Image(
            size_hint=(1, 0.6),
            allow_stretch=True,
            keep_ratio=True
        )
        self.layout.add_widget(self.video_widget)
        
        # Status label
        self.status_label = Label(
            size_hint=(1, 0.05),
            text="Initializing...",
            color=(1, 1, 0, 1),
            font_size='16sp'
        )
        self.layout.add_widget(self.status_label)
        
        # Instructions label
        self.instructions_label = Label(
            size_hint=(1, 0.03),
            text="",
            color=(0, 1, 0, 1),
            font_size='14sp'
        )
        self.layout.add_widget(self.instructions_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        
        self.btn_download = Button(
            text='📥 Download Sample Video',
            font_size='16sp',
            background_color=(0.3, 0.6, 0.9, 1),
            size_hint_x=0.34
        )
        self.btn_download.bind(on_press=self.download_sample_video)
        button_layout.add_widget(self.btn_download)
        
        self.btn_webcam = Button(
            text='📷 Use Webcam',
            font_size='16sp',
            background_color=(0.3, 0.8, 0.3, 1),
            size_hint_x=0.33
        )
        self.btn_webcam.bind(on_press=self.use_webcam)
        button_layout.add_widget(self.btn_webcam)
        
        self.btn_stop = Button(
            text='⏹ Stop',
            font_size='16sp',
            background_color=(0.8, 0.3, 0.3, 1),
            disabled=True,
            size_hint_x=0.33
        )
        self.btn_stop.bind(on_press=self.stop_processing)
        button_layout.add_widget(self.btn_stop)
        
        self.layout.add_widget(button_layout)
        
        # Processing control buttons
        process_layout = BoxLayout(size_hint=(1, 0.08), spacing=10)
        
        self.btn_start = Button(
            text='▶ Start Processing',
            font_size='16sp',
            background_color=(0.5, 0.5, 0.5, 1),
            disabled=True,
            size_hint_x=0.5
        )
        self.btn_start.bind(on_press=self.start_processing)
        process_layout.add_widget(self.btn_start)
        
        self.btn_reset = Button(
            text='🔄 Reset Selection',
            font_size='16sp',
            background_color=(0.5, 0.5, 0.5, 1),
            disabled=True,
            size_hint_x=0.5
        )
        self.btn_reset.bind(on_press=self.reset_selection)
        process_layout.add_widget(self.btn_reset)
        
        self.layout.add_widget(process_layout)
        
        # Bind touch events
        self.video_widget.bind(on_touch_down=self.on_video_touch)
        
        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0/30.0)
        
        # Show initial message
        self.status_label.text = "Ready - Click 'Download Sample Video' or 'Use Webcam'"
        
        return self.layout
    
    def download_sample_video(self, instance):
        """Download sample video"""
        self.status_label.text = "Downloading sample video..."
        self.status_label.color = (1, 1, 0, 1)
        
        # Run download in thread
        thread = threading.Thread(target=self._download_video)
        thread.daemon = True
        thread.start()
    
    def _download_video(self):
        """Download video in background"""
        try:
            video_path = "sample_walking.mp4"
            
            if not os.path.exists(video_path):
                # Sample video URL
                url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
                
                # Download with progress
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, int(downloaded * 100 / total_size))
                    self.status_label.text = f"Downloading: {percent}%"
                
                urllib.request.urlretrieve(url, video_path, report_progress)
            
            # Load the video
            self.video_path = video_path
            self.load_video_source()
            
        except Exception as e:
            self.status_label.text = f"Download failed: {str(e)}"
            self.status_label.color = (1, 0, 0, 1)
    
    def use_webcam(self, instance):
        """Use webcam as video source"""
        self.status_label.text = "Opening webcam..."
        
        # Try to open webcam
        self.capture = cv2.VideoCapture(0)
        
        if self.capture.isOpened():
            # Set webcam properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Get first frame
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                
                # Enable controls
                self.btn_start.disabled = False
                self.btn_start.background_color = (0, 1, 0, 1)
                self.btn_download.disabled = True
                self.btn_webcam.disabled = True
                self.btn_stop.disabled = False
                
                self.status_label.text = "✓ Webcam ready - Click 'Start Processing'"
                self.status_label.color = (0, 1, 0, 1)
                self.instructions_label.text = "Then tap on any object to track"
        else:
            self.status_label.text = "✗ No webcam found"
            self.status_label.color = (1, 0, 0, 1)
    
    def load_video_source(self):
        """Load video from file"""
        try:
            # Release previous capture
            if self.capture:
                self.capture.release()
            
            # Open video file
            self.capture = cv2.VideoCapture(self.video_path)
            
            if self.capture.isOpened():
                # Get first frame
                ret, frame = self.capture.read()
                if ret:
                    self.current_frame = frame
                    self.display_frame(frame)
                    
                    # Enable controls
                    self.btn_start.disabled = False
                    self.btn_start.background_color = (0, 1, 0, 1)
                    self.btn_download.disabled = True
                    self.btn_webcam.disabled = True
                    self.btn_stop.disabled = False
                    
                    self.status_label.text = "✓ Video loaded - Click 'Start Processing'"
                    self.status_label.color = (0, 1, 0, 1)
                    self.instructions_label.text = "Then tap on any object to track"
                    
        except Exception as e:
            self.status_label.text = f"Error loading video: {str(e)}"
            self.status_label.color = (1, 0, 0, 1)
    
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
        self.btn_reset.disabled = False
        self.btn_reset.background_color = (1, 0.5, 0, 1)
        
        self.status_label.text = "▶ Processing - Tap on an object to track"
        self.status_label.color = (1, 1, 0, 1)
        self.instructions_label.text = "👆 Tap on any object in the video"
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_processing(self, instance):
        """Stop video/processing"""
        self.processing = False
        self.tracking_active = False
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Update UI
        self.btn_start.disabled = True
        self.btn_start.background_color = (0.5, 0.5, 0.5, 1)
        self.btn_reset.disabled = True
        self.btn_reset.background_color = (0.5, 0.5, 0.5, 1)
        self.btn_download.disabled = False
        self.btn_webcam.disabled = False
        self.btn_stop.disabled = True
        
        self.status_label.text = "⏹ Stopped - Choose video source"
        self.status_label.color = (1, 1, 1, 1)
        self.instructions_label.text = ""
        
        # Clear video display
        self.video_widget.texture = None
    
    def reset_selection(self, instance):
        """Reset current object selection"""
        self.tracking_active = False
        self.selected_object = None
        self.status_label.text = "Selection reset - Tap new object to track"
        self.instructions_label.text = "👆 Tap on any object to start tracking"
    
    def on_video_touch(self, instance, touch):
        """Handle touch events for object selection"""
        if not self.processing or self.current_frame is None:
            return False
            
        # Check if touch is within video widget
        if instance.collide_point(touch.x, touch.y):
            # Convert coordinates
            h, w = self.current_frame.shape[:2]
            img_x = int((touch.x / instance.width) * w)
            img_y = int(((instance.height - touch.y) / instance.height) * h)
            
            # Ensure coordinates are valid
            img_x = max(0, min(img_x, w-1))
            img_y = max(0, min(img_y, h-1))
            
            # Select object
            self.select_object_at_position(img_x, img_y)
            return True
        return False
    
    def select_object_at_position(self, x, y):
        """Select object at clicked position"""
        try:
            # Detect objects in current frame
            objects = self.detector.detect(self.current_frame)
            
            # Find object containing click point
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Extract features
                    roi = self.current_frame[y1:y2, x1:x2]
                    features = self.extract_features(roi)
                    
                    if features is not None:
                        self.selected_object = {
                            'bbox': [x1, y1, x2, y2],
                            'features': features
                        }
                        self.tracking_active = True
                        
                        self.status_label.text = f"✓ Tracking object"
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
    
    def process_video(self):
        """Process video frames in thread"""
        while self.processing and self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                # Loop video for file, stop for webcam
                if self.video_path:  # It's a file
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:  # It's webcam
                    break
            
            # Process frame
            if self.tracking_active and self.selected_object:
                frame = self.process_frame(frame)
            
            # Add to queue
            if not self.result_queue.full():
                self.result_queue.put(frame)
            
            # Small delay to control processing speed
            time.sleep(0.01)
    
    def process_frame(self, frame):
        """Process a single frame"""
        result = frame.copy()
        
        try:
            # Detect objects
            objects = self.detector.detect(frame)
            
            # Find best match for selected object
            best_match = None
            best_score = 0
            best_bbox = None
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                roi = frame[y1:y2, x1:x2]
                features = self.extract_features(roi)
                
                if features is not None:
                    # Calculate feature similarity
                    score = self.match_features(features, self.selected_object['features'])
                    
                    if score > 0.3 and score > best_score:
                        best_score = score
                        best_match = obj
                        best_bbox = [x1, y1, x2, y2]
            
            if best_match:
                # Update selected object
                self.selected_object['bbox'] = best_bbox
                
                # Apply effects
                result = self.apply_background_blur(frame, best_bbox)
                result = self.highlight_object(result, best_bbox)
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
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are valid
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            if x2 > x1 and y2 > y1:
                # Create mask
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Apply blur
                blurred = cv2.GaussianBlur(frame, (51, 51), 30)
                
                # Combine using mask
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
                return result
            
        except Exception as e:
            print(f"Blur error: {e}")
        
        return frame
    
    def highlight_object(self, frame, bbox):
        """Highlight tracked object"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw simple focus indicator
            cv2.putText(frame, "FOCUS", (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Highlight error: {e}")
        
        return frame
    
    def display_frame(self, frame):
        """Display frame in Kivy widget"""
        if frame is None:
            return
            
        try:
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
        elif self.capture and not self.processing:
            # Show frames when not processing
            ret, frame = self.capture.read()
            if ret:
                self.display_frame(frame)
                self.current_frame = frame
    
    def on_stop(self):
        """Cleanup on exit"""
        self.processing = False
        if self.capture:
            self.capture.release()


class SimpleDetector:
    """Simple motion-based detector"""
    def __init__(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2()
        
    def detect(self, frame):
        """Detect objects in frame"""
        objects = []
        
        try:
            # Apply background subtraction
            fg_mask = self.back_sub.apply(frame)
            
            # Clean mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    objects.append({
                        'bbox': [x, y, x+w, y+h],
                        'area': area
                    })
                    
        except Exception as e:
            print(f"Detection error: {e}")
        
        return objects


if __name__ == '__main__':
    SmartFocusApp().run()