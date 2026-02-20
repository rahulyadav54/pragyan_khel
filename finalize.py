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
from pathlib import Path

class SmartFocusApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Video processing variables
        self.capture = None
        self.current_frame = None
        self.processing = False
        self.tracking_active = False
        self.result_queue = Queue(maxsize=2)
        self.video_path = None
        self.video_thread = None
        self.frame_count = 0
        
        # Tracking variables
        self.tracker = None
        self.tracker_bbox = None
        self.lost_counter = 0
        self.max_lost = 10
        self.selection_mode = False
        self.selected_point = None
        
    def build(self):
        """Build the UI"""
        Window.size = (1200, 800)
        Window.clearcolor = (0.15, 0.15, 0.15, 1)
        
        # Main layout
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Title
        title = Label(
            text="🎯 SmartFocus Pro Tracker",
            font_size='28sp',
            color=(0.2, 0.8, 1, 1),
            bold=True,
            size_hint=(1, 0.08)
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
            text="✅ Ready",
            color=(0, 1, 0, 1),
            font_size='18sp',
            size_hint=(1, 0.05)
        )
        self.layout.add_widget(self.status_label)
        
        # Instructions label
        self.instructions_label = Label(
            text="Click 'Start Webcam' or 'Load Video' to begin",
            color=(1, 1, 1, 1),
            font_size='16sp',
            size_hint=(1, 0.05)
        )
        self.layout.add_widget(self.instructions_label)
        
        # Control buttons
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        
        self.btn_webcam = Button(
            text='📷 Start Webcam',
            font_size='18sp',
            background_color=(0.2, 0.7, 0.2, 1),
            size_hint_x=0.25
        )
        self.btn_webcam.bind(on_press=self.start_webcam)
        button_layout.add_widget(self.btn_webcam)
        
        self.btn_video = Button(
            text='📥 Load Video',
            font_size='18sp',
            background_color=(0.2, 0.5, 0.9, 1),
            size_hint_x=0.25
        )
        self.btn_video.bind(on_press=self.load_video)
        button_layout.add_widget(self.btn_video)
        
        self.btn_focus = Button(
            text='🎯 Focus Mode',
            font_size='18sp',
            background_color=(0.8, 0.5, 0.2, 1),
            size_hint_x=0.25
        )
        self.btn_focus.bind(on_press=self.toggle_focus_mode)
        button_layout.add_widget(self.btn_focus)
        
        self.btn_stop = Button(
            text='⏹ Stop',
            font_size='18sp',
            background_color=(0.8, 0.2, 0.2, 1),
            disabled=True,
            size_hint_x=0.25
        )
        self.btn_stop.bind(on_press=self.stop_processing)
        button_layout.add_widget(self.btn_stop)
        
        self.layout.add_widget(button_layout)
        
        # Bind touch events
        self.video_widget.bind(on_touch_down=self.on_video_touch)
        
        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0/30.0)
        
        self.focus_mode = False
        
        return self.layout
    
    def start_webcam(self, instance):
        """Start webcam"""
        self.status_label.text = "📷 Starting webcam..."
        
        # Try to open webcam
        self.capture = cv2.VideoCapture(0)
        
        if self.capture.isOpened():
            # Set properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Update UI
            self.btn_webcam.disabled = True
            self.btn_video.disabled = True
            self.btn_stop.disabled = False
            
            self.status_label.text = "✅ Webcam running"
            self.instructions_label.text = "Click 'Focus Mode' then tap an object to track"
            
            # Start processing
            self.processing = True
            self.video_thread = threading.Thread(target=self.process_video)
            self.video_thread.daemon = True
            self.video_thread.start()
        else:
            self.status_label.text = "❌ Could not open webcam"
    
    def load_video(self, instance):
        """Load sample video"""
        self.status_label.text = "📥 Loading video..."
        
        # Create videos directory
        video_dir = os.path.join(str(Path.home()), "SmartFocusVideos")
        os.makedirs(video_dir, exist_ok=True)
        
        video_path = os.path.join(video_dir, "sample.mp4")
        
        # Download if not exists
        if not os.path.exists(video_path):
            try:
                url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/tracking/tracking_benchmark/BlurBody/BlurBody.mp4"
                urllib.request.urlretrieve(url, video_path)
            except:
                # Create test video if download fails
                self.create_test_video(video_path)
        
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        
        if self.capture.isOpened():
            self.btn_webcam.disabled = True
            self.btn_video.disabled = True
            self.btn_stop.disabled = False
            
            self.status_label.text = "✅ Video loaded"
            self.instructions_label.text = "Click 'Focus Mode' then tap an object to track"
            
            self.processing = True
            self.video_thread = threading.Thread(target=self.process_video)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def create_test_video(self, path):
        """Create test video with moving objects"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 30, (640, 480))
        
        for i in range(300):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)
            
            # Moving ball
            x = int(320 + 200 * np.sin(i * 0.05))
            y = int(240 + 150 * np.cos(i * 0.03))
            cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
            
            out.write(frame)
        
        out.release()
    
    def toggle_focus_mode(self, instance):
        """Toggle focus mode"""
        self.focus_mode = not self.focus_mode
        
        if self.focus_mode:
            self.btn_focus.background_color = (1, 0.8, 0, 1)
            self.instructions_label.text = "Tap on an object to focus and blur background"
            self.status_label.text = "🎯 Focus Mode ON - Tap object"
        else:
            self.btn_focus.background_color = (0.8, 0.5, 0.2, 1)
            self.instructions_label.text = "Focus Mode OFF"
            self.status_label.text = "✅ Normal mode"
            self.tracking_active = False
            self.tracker = None
    
    def stop_processing(self, instance):
        """Stop processing"""
        self.processing = False
        self.tracking_active = False
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        self.btn_webcam.disabled = False
        self.btn_video.disabled = False
        self.btn_stop.disabled = True
        self.focus_mode = False
        self.btn_focus.background_color = (0.8, 0.5, 0.2, 1)
        
        self.status_label.text = "⏹ Stopped"
        self.instructions_label.text = "Click 'Start Webcam' or 'Load Video' to begin"
        self.video_widget.texture = None
    
    def on_video_touch(self, instance, touch):
        """Handle touch on video"""
        if not self.processing or not self.focus_mode or self.current_frame is None:
            return False
        
        if instance.collide_point(touch.x, touch.y):
            # Get image dimensions
            h, w = self.current_frame.shape[:2]
            
            # Get widget dimensions
            widget_w, widget_h = instance.size
            
            # Calculate scaling
            scale_x = w / widget_w
            scale_y = h / widget_h
            
            # Convert coordinates
            x = int(touch.x * scale_x)
            y = int((widget_h - touch.y) * scale_y)
            
            # Ensure within bounds
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            self.selected_point = (x, y)
            self.start_tracking(x, y)
            
            return True
        return False
    
    def start_tracking(self, x, y):
        """Start tracking at selected point"""
        try:
            # Create initial bounding box around click
            box_size = 50
            x1 = max(0, x - box_size)
            y1 = max(0, y - box_size)
            x2 = min(self.current_frame.shape[1], x + box_size)
            y2 = min(self.current_frame.shape[0], y + box_size)
            
            # Create tracker (use CSRT for better accuracy)
            self.tracker = cv2.TrackerCSRT_create()
            
            # Initialize tracker with current frame and bounding box
            bbox = (x1, y1, x2 - x1, y2 - y1)
            self.tracker.init(self.current_frame, bbox)
            
            # Store bounding box
            self.tracker_bbox = [int(x1), int(y1), int(x2), int(y2)]
            self.tracking_active = True
            self.lost_counter = 0
            
            self.status_label.text = "✅ Tracking object"
            self.instructions_label.text = "Object locked - background blurred"
            
        except Exception as e:
            print(f"Tracking error: {e}")
            self.status_label.text = "❌ Could not track object"
    
    def apply_focus_effect(self, frame, bbox):
        """Apply focus effect - blur background, keep object sharp"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure valid coordinates
            x1 = max(0, min(x1, frame.shape[1]-1))
            y1 = max(0, min(y1, frame.shape[0]-1))
            x2 = max(x1+1, min(x2, frame.shape[1]))
            y2 = max(y1+1, min(y2, frame.shape[0]))
            
            # Create a copy for the result
            result = frame.copy()
            
            # Apply strong blur to entire frame
            blurred = cv2.GaussianBlur(frame, (99, 99), 50)
            
            # Copy the sharp object region from original frame
            result[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            
            # Create a feathered mask for smooth transition
            mask = np.zeros(frame.shape[:2], dtype=np.float32)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
            
            # Feather the mask
            mask = cv2.GaussianBlur(mask, (51, 51), 25)
            mask = np.stack([mask] * 3, axis=2)
            
            # Blend sharp and blurred images using mask
            result = (frame * mask + blurred * (1 - mask)).astype(np.uint8)
            
            # Draw highlight box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw corner markers
            corner_len = 20
            cv2.line(result, (x1, y1), (x1+corner_len, y1), (0, 255, 0), 2)
            cv2.line(result, (x1, y1), (x1, y1+corner_len), (0, 255, 0), 2)
            cv2.line(result, (x2, y1), (x2-corner_len, y1), (0, 255, 0), 2)
            cv2.line(result, (x2, y1), (x2, y1+corner_len), (0, 255, 0), 2)
            cv2.line(result, (x1, y2), (x1+corner_len, y2), (0, 255, 0), 2)
            cv2.line(result, (x1, y2), (x1, y2-corner_len), (0, 255, 0), 2)
            cv2.line(result, (x2, y2), (x2-corner_len, y2), (0, 255, 0), 2)
            cv2.line(result, (x2, y2), (x2, y2-corner_len), (0, 255, 0), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(result, (center_x, center_y), 5, (0, 255, 0), -1)
            
            return result
            
        except Exception as e:
            print(f"Focus effect error: {e}")
            return frame
    
    def process_video(self):
        """Process video frames"""
        while self.processing and self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            
            if not ret:
                # Loop video if it's a file
                if self.video_path:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            self.frame_count += 1
            self.current_frame = frame.copy()
            
            # Apply tracking if active
            if self.tracking_active and self.tracker is not None:
                try:
                    # Update tracker
                    success, bbox = self.tracker.update(frame)
                    
                    if success:
                        # Get bounding box coordinates
                        x, y, w, h = [int(v) for v in bbox]
                        self.tracker_bbox = [x, y, x + w, y + h]
                        self.lost_counter = 0
                        
                        # Apply focus effect
                        frame = self.apply_focus_effect(frame, self.tracker_bbox)
                        
                    else:
                        self.lost_counter += 1
                        
                        if self.lost_counter < self.max_lost and self.tracker_bbox:
                            # Use last known position
                            frame = self.apply_focus_effect(frame, self.tracker_bbox)
                            cv2.putText(frame, "SEARCHING...", (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                        else:
                            self.tracking_active = False
                            self.tracker = None
                            cv2.putText(frame, "OBJECT LOST - Tap again", (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                except Exception as e:
                    print(f"Tracker update error: {e}")
                    self.tracking_active = False
                    self.tracker = None
            
            # Add frame info
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Update queue
            if not self.result_queue.full():
                self.result_queue.put(frame)
    
    def update_frame(self, dt):
        """Update display"""
        if not self.result_queue.empty():
            frame = self.result_queue.get()
            self.display_frame(frame)
    
    def display_frame(self, frame):
        """Display frame"""
        if frame is None:
            return
        
        try:
            # Convert to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.video_widget.texture = texture
        except Exception as e:
            print(f"Display error: {e}")
    
    def on_stop(self):
        """Cleanup"""
        self.processing = False
        if self.capture:
            self.capture.release()

if __name__ == '__main__':
    SmartFocusApp().run()