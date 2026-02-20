"""
AI-Powered Smart Focus System - GUI Version
With video selection and start processing button
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import logging
import sys
import os
from collections import deque
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartFocusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Smart Focus System")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Video processing variables
        self.cap = None
        self.is_processing = False
        self.process_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.command_queue = queue.Queue()
        
        # Selected object tracking
        self.selected_object = None
        self.tracking_active = False
        self.tracker = None
        self.click_point = None
        self.last_known_bbox = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Bind mouse click for object selection
        self.video_label.bind("<Button-1>", self.on_video_click)
        
        logger.info("GUI initialized")
    
    def create_widgets(self):
        """Create all GUI elements"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Video source selection
        ttk.Label(control_frame, text="Video Source:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.video_source = tk.StringVar(value="webcam")
        ttk.Radiobutton(control_frame, text="Webcam", variable=self.video_source, 
                       value="webcam").grid(row=1, column=0, sticky=tk.W, padx=20)
        ttk.Radiobutton(control_frame, text="Sample Video", variable=self.video_source, 
                       value="file").grid(row=2, column=0, sticky=tk.W, padx=20)
        
        # File selection
        self.file_path = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.file_path, width=30).grid(row=3, column=0, pady=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=3, column=1, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Processing controls
        ttk.Label(control_frame, text="Processing Controls:", font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="▶ Start Processing", 
                                      command=self.start_processing, width=20)
        self.start_button.grid(row=6, column=0, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="■ Stop Processing", 
                                     command=self.stop_processing, width=20, state=tk.DISABLED)
        self.stop_button.grid(row=7, column=0, pady=5)
        
        # Settings
        ttk.Label(control_frame, text="Settings:", font=('Arial', 10, 'bold')).grid(row=8, column=0, sticky=tk.W, pady=5)
        
        # Blur strength
        ttk.Label(control_frame, text="Blur Strength:").grid(row=9, column=0, sticky=tk.W)
        self.blur_strength = tk.IntVar(value=45)
        ttk.Scale(control_frame, from_=15, to=99, variable=self.blur_strength, 
                 orient=tk.HORIZONTAL, length=150).grid(row=10, column=0, pady=2)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence Threshold:").grid(row=11, column=0, sticky=tk.W)
        self.confidence = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.confidence, 
                 orient=tk.HORIZONTAL, length=150).grid(row=12, column=0, pady=2)
        
        # Focus radius
        ttk.Label(control_frame, text="Focus Radius:").grid(row=13, column=0, sticky=tk.W)
        self.focus_radius = tk.DoubleVar(value=1.5)
        ttk.Scale(control_frame, from_=1.0, to=3.0, variable=self.focus_radius, 
                 orient=tk.HORIZONTAL, length=150).grid(row=14, column=0, pady=2)
        
        # Status display
        ttk.Label(control_frame, text="Status:", font=('Arial', 10, 'bold')).grid(row=15, column=0, sticky=tk.W, pady=5)
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=16, column=0, sticky=tk.W)
        
        # Selected object info
        self.object_label = ttk.Label(control_frame, text="No object selected", foreground="gray")
        self.object_label.grid(row=17, column=0, sticky=tk.W, pady=5)
        
        # Instructions
        instructions = """
Instructions:
1. Select video source
2. Click 'Start Processing'
3. Click on any object to select it
4. Selected object stays in focus
5. Background becomes blurred
6. Click new object to switch focus
7. Press 'q' to quit
        """
        ttk.Label(control_frame, text=instructions, justify=tk.LEFT, 
                 foreground="gray").grid(row=18, column=0, pady=10)
        
        # Right panel - Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Display", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Video label for displaying frames
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
        # FPS display
        self.fps_label = ttk.Label(video_frame, text="FPS: 0.0", foreground="green")
        self.fps_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def browse_file(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
            logger.info(f"Selected file: {filename}")
    
    def on_video_click(self, event):
        """Handle mouse click on video for object selection"""
        if self.is_processing and self.cap is not None:
            # Get click coordinates relative to video display
            x = event.x
            y = event.y
            
            # Scale coordinates if video is resized
            if hasattr(self, 'display_width') and hasattr(self, 'display_height'):
                if self.display_width > 0 and self.display_height > 0:
                    scale_x = self.frame_width / self.display_width
                    scale_y = self.frame_height / self.display_height
                    x = int(x * scale_x)
                    y = int(y * scale_y)
            
            self.click_point = (x, y)
            logger.info(f"Clicked at ({x}, {y}) - Selecting object")
            self.status_label.config(text="Selecting object...", foreground="orange")
    
    def start_processing(self):
        """Start video processing"""
        if self.is_processing:
            return
        
        # Get video source
        source = self.video_source.get()
        
        if source == "webcam":
            video_path = 0  # Webcam
            self.status_label.config(text="Opening webcam...", foreground="blue")
        else:
            video_path = self.file_path.get()
            if not video_path:
                messagebox.showerror("Error", "Please select a video file first!")
                return
            if not os.path.exists(video_path):
                messagebox.showerror("Error", "Video file not found!")
                return
            self.status_label.config(text=f"Opening video: {os.path.basename(video_path)}", foreground="blue")
        
        # Open video capture
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video source!")
                return
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video opened: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
            
            # Update UI
            self.is_processing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Processing... Click on objects to select", foreground="green")
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_video, daemon=True)
            self.process_thread.start()
            
            # Start GUI update loop
            self.update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start processing: {str(e)}")
            logger.error(f"Start processing error: {e}")
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped", foreground="red")
        self.object_label.config(text="No object selected", foreground="gray")
        self.fps_label.config(text="FPS: 0.0")
        
        # Clear display
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.display_image(blank_image)
        
        logger.info("Processing stopped")
    
    def process_video(self):
        """Video processing thread"""
        fps_history = deque(maxlen=30)
        
        # Try to import YOLO (optional)
        try:
            from ultralytics import YOLO
            detector = YOLO('yolov8n.pt')
            has_detector = True
            logger.info("YOLO detector loaded")
        except:
            has_detector = False
            logger.warning("YOLO not available - using fallback detection")
        
        while self.is_processing and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video for file sources
                    if self.video_source.get() == "file":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Calculate FPS
                current_time = time.time()
                if hasattr(self, 'prev_time'):
                    fps = 1.0 / (current_time - self.prev_time)
                    fps_history.append(fps)
                self.prev_time = current_time
                
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                
                # Get current settings
                blur_strength = self.blur_strength.get()
                if blur_strength % 2 == 0:
                    blur_strength += 1
                confidence = self.confidence.get()
                focus_radius = self.focus_radius.get()
                
                # Detect objects (optional)
                detections = []
                if has_detector:
                    try:
                        results = detector(frame, verbose=False)[0]
                        if results.boxes is not None:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            confs = results.boxes.conf.cpu().numpy()
                            classes = results.boxes.cls.cpu().numpy().astype(int)
                            
                            for box, conf, cls in zip(boxes, confs, classes):
                                if conf >= confidence:
                                    detections.append({
                                        'bbox': box.tolist(),
                                        'label': detector.names[cls],
                                        'confidence': float(conf)
                                    })
                    except:
                        pass
                
                # Handle object selection from click
                if self.click_point is not None and detections:
                    x, y = self.click_point
                    for det in detections:
                        bbox = det['bbox']
                        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                            self.selected_object = det
                            self.tracking_active = True
                            self.last_known_bbox = det['bbox']
                            
                            # Initialize tracker
                            self.tracker = cv2.TrackerCSRT_create()
                            tx, ty, tw, th = det['bbox'][0], det['bbox'][1], \
                                            det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1]
                            self.tracker.init(frame, (tx, ty, tw, th))
                            
                            logger.info(f"Selected: {det['label']}")
                            self.root.after(0, self.update_object_label, det['label'])
                            break
                    self.click_point = None
                
                # Apply tracking and focus effect
                result_frame = frame.copy()
                
                if self.tracking_active and self.selected_object is not None:
                    # Update tracker
                    if self.tracker is not None:
                        success, bbox = self.tracker.update(frame)
                        if success:
                            x, y, w, h = bbox
                            self.last_known_bbox = [x, y, x+w, y+h]
                            self.selected_object['bbox'] = self.last_known_bbox
                    
                    # Apply focus effect if we have a bbox
                    if self.last_known_bbox is not None:
                        result_frame = self.apply_focus_effect(
                            frame, self.last_known_bbox, blur_strength, focus_radius
                        )
                        
                        # Draw tracking box
                        bbox = self.last_known_bbox
                        cv2.rectangle(result_frame, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    (0, 255, 0), 3)
                        
                        # Draw label
                        if 'label' in self.selected_object:
                            label = f"{self.selected_object['label']} ({self.selected_object.get('confidence', 1.0):.2f})"
                            cv2.putText(result_frame, label, 
                                      (int(bbox[0]), int(bbox[1])-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw detections if no object selected
                elif detections and not self.tracking_active:
                    for det in detections:
                        bbox = det['bbox']
                        cv2.rectangle(result_frame, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    (255, 255, 0), 2)
                        
                        label = f"{det['label']} ({det['confidence']:.2f})"
                        cv2.putText(result_frame, label,
                                  (int(bbox[0]), int(bbox[1])-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Add FPS and info
                cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                status = "Tracking" if self.tracking_active else "Click to select"
                cv2.putText(result_frame, f"Status: {status}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Put frame in queue for GUI display
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(result_frame)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
    
    def apply_focus_effect(self, frame, bbox, blur_strength, focus_radius):
        """Apply focus effect with background blur"""
        # Create blurred version
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        # Create circular mask
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = int(max(x2 - x1, y2 - y1) * focus_radius / 2)
        
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Apply blur with mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (frame * mask_3channel + blurred * (1 - mask_3channel)).astype(np.uint8)
        
        return result
    
    def update_frame(self):
        """Update GUI with new frame"""
        if not self.is_processing:
            return
        
        try:
            # Get frame from queue
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # Display frame
                self.display_image(frame)
                
                # Update FPS in status
                if hasattr(self, 'prev_gui_time'):
                    gui_fps = 1.0 / (time.time() - self.prev_gui_time)
                    self.fps_label.config(text=f"FPS: {gui_fps:.1f}")
                self.prev_gui_time = time.time()
            
            # Schedule next update
            self.root.after(30, self.update_frame)
            
        except Exception as e:
            logger.error(f"GUI update error: {e}")
    
    def display_image(self, frame):
        """Display frame in GUI"""
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        display_width = 800
        display_height = 600
        self.display_width = display_width
        self.display_height = display_height
        
        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)
        
        # Update label
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep reference
    
    def update_object_label(self, label):
        """Update object selection label"""
        self.object_label.config(text=f"Selected: {label}", foreground="green")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_processing()
        self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = SmartFocusGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()