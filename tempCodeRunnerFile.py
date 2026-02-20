#!/usr/bin/env python3
"""
AI-Powered Smart Focus System
Complete working prototype for real-time object tracking and background blur
"""

import cv2
import numpy as np
import torch
import argparse
import time
import logging
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """System configuration"""
    # Input/Output
    input_source: str = "0"  # Webcam or video file
    output_path: Optional[str] = None
    save_output: bool = False
    
    # Model settings
    model_path: str = "yolov8n.pt"  # Will download automatically if not present
    use_cuda: bool = torch.cuda.is_available()
    confidence_threshold: float = 0.5
    
    # Tracking settings
    max_tracking_loss: int = 30  # Frames to keep tracking after loss
    use_optical_flow: bool = True
    use_kalman: bool = True
    
    # Effect settings
    blur_strength: int = 45  # Must be odd number
    focus_radius_factor: float = 1.5
    highlight_alpha: float = 0.3  # Transparency for highlight
    
    # Performance
    frame_skip: int = 1  # Process every Nth frame
    resize_width: Optional[int] = 640  # Resize for faster processing
    show_fps: bool = True
    
    def __post_init__(self):
        if self.blur_strength % 2 == 0:
            self.blur_strength += 1

# ============================================================================
# UTILITY CLASSES
# ============================================================================

class FPSMonitor:
    """FPS monitoring utility"""
    def __init__(self, avg_frames=30):
        self.fps_history = deque(maxlen=avg_frames)
        self.prev_time = time.time()
        self.fps = 0
    
    def update(self):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time)
        self.fps_history.append(self.fps)
        self.prev_time = current_time
    
    def get_fps(self):
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

class ClickHandler:
    """Handle mouse clicks for object selection"""
    def __init__(self):
        self.click_point = None
        self.window_name = "Smart Focus System"
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            logger.info(f"Clicked at ({x}, {y})")
    
    def get_click(self):
        point = self.click_point
        self.click_point = None
        return point
    
    def clear_click(self):
        self.click_point = None

# ============================================================================
# OBJECT DETECTOR (YOLOv8)
# ============================================================================

class ObjectDetector:
    """YOLOv8 object detector"""
    def __init__(self, config: Config):
        self.config = config
        self.device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
        
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model on {self.device}")
            
            # Download model if not exists
            model_path = config.model_path
            if not os.path.exists(model_path):
                logger.info(f"Model not found at {model_path}, downloading...")
                # Will download from ultralytics hub
                self.model = YOLO('yolov8n.pt')
                # Save for future use
                self.model.export(format='onnx')  # Optional
            else:
                self.model = YOLO(model_path)
            
            if self.device == 'cuda':
                self.model.to('cuda')
            
            # Class names
            self.class_names = self.model.names
            logger.info(f"Model loaded with {len(self.class_names)} classes")
            
        except ImportError:
            logger.warning("Ultralytics not installed. Using fallback detector.")
            self.model = None
            self.class_names = {0: 'object'}
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame"""
        if self.model is None:
            # Fallback: return dummy detection for center of frame
            h, w = frame.shape[:2]
            return [{
                'id': 0,
                'bbox': [w//4, h//4, 3*w//4, 3*h//4],
                'label': 'object',
                'confidence': 1.0,
                'class_id': 0
            }]
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            detections = []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf >= self.config.confidence_threshold:
                        detections.append({
                            'id': i,
                            'bbox': box.tolist(),
                            'label': self.class_names[class_id],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

# ============================================================================
# OBJECT TRACKER
# ============================================================================

class ObjectTracker:
    """Multi-method object tracker with CSRT + Optical Flow + Kalman"""
    def __init__(self, config: Config):
        self.config = config
        self.tracker = None
        self.loss_counter = 0
        self.last_bbox = None
        self.last_frame = None
        self.tracking_points = None
        self.initialized = False
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.03
    
    def init_tracker(self, frame: np.ndarray, bbox: List[float]):
        """Initialize tracker with first frame and bbox"""
        try:
            # CSRT tracker
            self.tracker = cv2.TrackerCSRT.create()
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            self.tracker.init(frame, (x, y, w, h))
            
            # Store data
            self.last_frame = frame.copy()
            self.last_bbox = bbox.copy()
            self.loss_counter = 0
            
            # Initialize Kalman
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            self.kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
            
            # Extract features for optical flow
            if self.config.use_optical_flow:
                mask = np.zeros(frame.shape[:2], np.uint8)
                mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
                self.tracking_points = cv2.goodFeaturesToTrack(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    mask=mask,
                    **self.feature_params
                )
            
            self.initialized = True
            logger.debug(f"Tracker initialized with bbox: {bbox}")
            
        except Exception as e:
            logger.error(f"Tracker initialization failed: {e}")
            self.initialized = False
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[List[float]]]:
        """Update tracker with new frame"""
        if not self.initialized or self.tracker is None:
            return False, None
        
        # Try CSRT tracking
        success, bbox = self.tracker.update(frame)
        
        if success:
            # Convert to [x1, y1, x2, y2]
            x, y, w, h = bbox
            bbox = [x, y, x + w, y + h]
            
            # Update Kalman
            if self.config.use_kalman:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                self.kalman.correct(np.array([[center_x], [center_y]], np.float32))
            
            self.last_bbox = bbox
            self.last_frame = frame.copy()
            self.loss_counter = 0
            
            # Update optical flow points
            if self.config.use_optical_flow:
                self._update_optical_flow_points(frame, bbox)
            
            return True, bbox
        
        else:
            self.loss_counter += 1
            
            if self.loss_counter < self.config.max_tracking_loss:
                # Try optical flow
                if self.config.use_optical_flow:
                    flow_bbox = self._track_with_optical_flow(frame)
                    if flow_bbox is not None:
                        bbox = flow_bbox
                        
                        # Apply Kalman prediction
                        if self.config.use_kalman:
                            predicted = self.kalman.predict()
                            bbox = self._blend_predictions(bbox, predicted)
                        
                        self.last_bbox = bbox
                        self.last_frame = frame.copy()
                        return True, bbox
            
            return False, None
    
    def _update_optical_flow_points(self, frame: np.ndarray, bbox: List[float]):
        """Update feature points for optical flow"""
        if self.tracking_points is not None and len(self.tracking_points) > 0:
            try:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    self.tracking_points,
                    None,
                    **self.lk_params
                )
                
                # Keep good points
                good_new = next_points[status == 1]
                if len(good_new) > 0:
                    self.tracking_points = good_new.reshape(-1, 1, 2)
                    
                    # Add new points if needed
                    if len(self.tracking_points) < self.feature_params['maxCorners']:
                        mask = np.zeros(frame.shape[:2], np.uint8)
                        mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
                        
                        new_points = cv2.goodFeaturesToTrack(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            mask=mask,
                            **self.feature_params
                        )
                        
                        if new_points is not None:
                            self.tracking_points = np.vstack((self.tracking_points, new_points))
                            
            except Exception as e:
                logger.debug(f"Optical flow update failed: {e}")
    
    def _track_with_optical_flow(self, frame: np.ndarray) -> Optional[List[float]]:
        """Track using optical flow when main tracker fails"""
        if self.tracking_points is None or len(self.tracking_points) < 4:
            return None
        
        try:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                self.tracking_points,
                None,
                **self.lk_params
            )
            
            good_new = next_points[status == 1]
            
            if len(good_new) < 4:
                return None
            
            # Calculate bbox from points
            x_coords = good_new[:, 0]
            y_coords = good_new[:, 1]
            
            x1 = max(0, np.min(x_coords) - 20)
            y1 = max(0, np.min(y_coords) - 20)
            x2 = min(frame.shape[1], np.max(x_coords) + 20)
            y2 = min(frame.shape[0], np.max(y_coords) + 20)
            
            return [x1, y1, x2, y2]
            
        except Exception as e:
            logger.debug(f"Optical flow tracking failed: {e}")
            return None
    
    def _blend_predictions(self, flow_bbox: List[float], kalman_pred: np.ndarray) -> List[float]:
        """Blend optical flow and Kalman predictions"""
        kalman_center = kalman_pred[:2].flatten()
        flow_center = [(flow_bbox[0] + flow_bbox[2]) / 2,
                       (flow_bbox[1] + flow_bbox[3]) / 2]
        
        # Weighted average
        alpha = 0.7  # Weight for optical flow
        blended_center = [
            alpha * flow_center[0] + (1 - alpha) * kalman_center[0],
            alpha * flow_center[1] + (1 - alpha) * kalman_center[1]
        ]
        
        # Maintain size from last known bbox
        w = self.last_bbox[2] - self.last_bbox[0]
        h = self.last_bbox[3] - self.last_bbox[1]
        
        return [
            blended_center[0] - w/2,
            blended_center[1] - h/2,
            blended_center[0] + w/2,
            blended_center[1] + h/2
        ]

# ============================================================================
# EFFECT RENDERER
# ============================================================================

class EffectRenderer:
    """Apply focus and blur effects"""
    def __init__(self, config: Config):
        self.config = config
        self.colors = {
            'focus': (0, 255, 0),      # Green
            'detection': (255, 255, 0), # Yellow
            'text': (255, 255, 255),    # White
            'highlight': (0, 255, 255)   # Cyan
        }
    
    def apply_focus_effect(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Apply focus effect with background blur"""
        # Create blurred version
        blurred = cv2.GaussianBlur(frame, (self.config.blur_strength, self.config.blur_strength), 0)
        
        # Create result frame
        result = frame.copy()
        
        # Get bbox coordinates
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Expand focus area
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = int(max(x2 - x1, y2 - y1) * self.config.focus_radius_factor / 2)
        
        # Create circular mask with falloff
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Smooth mask with falloff
        mask = np.clip(1 - (dist_from_center - radius) / radius, 0, 1)
        mask = np.clip(mask, 0, 1)
        mask = mask ** 2  # Sharper falloff
        
        # Apply effect
        for c in range(3):
            result[:, :, c] = (frame[:, :, c] * mask + blurred[:, :, c] * (1 - mask))
        
        return result.astype(np.uint8)
    
    def draw_tracking_info(self, frame: np.ndarray, bbox: List[float], object_info: dict):
        """Draw tracking information"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['focus'], 3)
        
        # Draw corners
        corner_length = 20
        corners = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2)
        ]
        for cx, cy in corners:
            cv2.circle(frame, (cx, cy), 5, self.colors['focus'], -1)
        
        # Draw label
        label = f"{object_info['label']} ({object_info['confidence']:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(frame,
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1),
                     self.colors['focus'],
                     -1)
        
        # Draw label text
        cv2.putText(frame, label,
                   (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   self.colors['text'],
                   2)
        
        # Draw center point
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, self.colors['highlight'], -1)
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw all detected objects"""
        result = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), self.colors['detection'], 2)
            
            # Draw label
            label = f"{detection['label']} ({detection['confidence']:.2f})"
            cv2.putText(result, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       self.colors['detection'],
                       2)
        
        return result
    
    def draw_fps(self, frame: np.ndarray, fps: float):
        """Draw FPS counter"""
        cv2.putText(frame, f"FPS: {fps:.1f}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   self.colors['text'],
                   2)
    
    def draw_instructions(self, frame: np.ndarray):
        """Draw instructions"""
        instructions = [
            "Click on object to select",
            "Press 'q' to quit",
            "Press 's' to save frame"
        ]
        
        y_offset = 60
        for instruction in instructions:
            cv2.putText(frame, instruction,
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       self.colors['text'],
                       1)
            y_offset += 25

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class SmartFocusSystem:
    """Main application class"""
    def __init__(self, config: Config):
        self.config = config
        self.selected_object = None
        self.tracking_active = False
        self.frame_count = 0
        self.process_every_n = config.frame_skip
        
        # Initialize components
        logger.info("Initializing Smart Focus System")
        logger.info(f"Device: {'CUDA' if config.use_cuda else 'CPU'}")
        
        self.detector = ObjectDetector(config)
        self.tracker = ObjectTracker(config)
        self.renderer = EffectRenderer(config)
        self.fps_monitor = FPSMonitor()
        self.click_handler = ClickHandler()
        
        # Setup video capture
        self._setup_capture()
        
        # Setup video writer
        if config.save_output and config.output_path:
            self._setup_writer()
    
    def _setup_capture(self):
        """Setup video capture"""
        source = self.config.input_source
        if source.isdigit():
            source = int(source)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Resize if needed
        if self.config.resize_width:
            scale = self.config.resize_width / self.frame_width
            self.process_height = int(self.frame_height * scale)
            self.process_width = self.config.resize_width
        else:
            self.process_width = self.frame_width
            self.process_height = self.frame_height
        
        logger.info(f"Input: {source}, Size: {self.frame_width}x{self.frame_height}")
        logger.info(f"Processing size: {self.process_width}x{self.process_height}")
    
    def _setup_writer(self):
        """Setup video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for processing if needed"""
        if self.config.resize_width:
            return cv2.resize(frame, (self.process_width, self.process_height))
        return frame
    
    def postprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize back to original size if needed"""
        if self.config.resize_width:
            return cv2.resize(frame, (self.frame_width, self.frame_height))
        return frame
    
    def find_object_at_point(self, detections: List[dict], point: Tuple[int, int]) -> Optional[dict]:
        """Find object at clicked point"""
        x, y = point
        
        # Adjust point if frame was resized
        if self.config.resize_width:
            scale_x = self.frame_width / self.process_width
            scale_y = self.frame_height / self.process_height
            x = int(x * scale_x)
            y = int(y * scale_y)
        
        for detection in detections:
            bbox = detection['bbox']
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                return detection
        
        return None
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def attempt_redetection(self, frame: np.ndarray, detections: List[dict]) -> bool:
        """Attempt to re-detect lost object"""
        if self.selected_object is None:
            return False
        
        best_match = None
        best_iou = 0.3
        
        for detection in detections:
            if detection['label'] == self.selected_object['label']:
                iou = self.calculate_iou(
                    self.selected_object['bbox'],
                    detection['bbox']
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = detection
        
        if best_match is not None:
            self.selected_object = best_match
            self.tracker.init_tracker(frame, best_match['bbox'])
            logger.info("Object re-detected successfully")
            return True
        
        return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        self.frame_count += 1
        process_this_frame = (self.frame_count % self.process_every_n == 0)
        
        # Resize for processing
        process_frame = self.preprocess_frame(frame) if process_this_frame else None
        
        if process_this_frame:
            # Detect objects
            detections = self.detector.detect(process_frame)
            
            # Handle user click
            click_point = self.click_handler.get_click()
            if click_point is not None:
                selected = self.find_object_at_point(detections, click_point)
                if selected is not None:
                    self.selected_object = selected
                    self.tracking_active = True
                    self.tracker.init_tracker(process_frame, selected['bbox'])
                    logger.info(f"Selected: {selected['label']}")
        
        # Prepare result frame (start with original)
        result_frame = frame.copy()
        
        # Apply tracking and effects
        if self.tracking_active and self.selected_object is not None:
            if process_this_frame:
                # Update tracker
                tracked, bbox = self.tracker.update(process_frame)
                
                if tracked:
                    self.selected_object['bbox'] = bbox
                    
                    # Scale bbox back to original size
                    if self.config.resize_width:
                        scale_x = self.frame_width / self.process_width
                        scale_y = self.frame_height / self.process_height
                        bbox_original = [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
                        ]
                    else:
                        bbox_original = bbox
                    
                    # Apply focus effect to original frame
                    result_frame = self.renderer.apply_focus_effect(frame, bbox_original)
                    self.renderer.draw_tracking_info(result_frame, bbox_original, self.selected_object)
                    
                else:
                    # Try re-detection
                    self.tracking_active = self.attempt_redetection(process_frame, detections)
                    if not self.tracking_active:
                        logger.debug("Tracking lost")
            
            else:
                # Use last known bbox for this frame
                if self.selected_object and 'bbox' in self.selected_object:
                    result_frame = self.renderer.apply_focus_effect(
                        frame, self.selected_object['bbox']
                    )
                    self.renderer.draw_tracking_info(
                        result_frame, 
                        self.selected_object['bbox'],
                        self.selected_object
                    )
        
        else:
            # No object selected - show detections
            if process_this_frame and 'detections' in locals():
                # Scale detections back to original size
                if self.config.resize_width:
                    scale_x = self.frame_width / self.process_width
                    scale_y = self.frame_height / self.process_height
                    for det in detections:
                        bbox = det['bbox']
                        det['bbox'] = [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
                        ]
                
                result_frame = self.renderer.draw_detections(frame, detections)
        
        # Draw overlays
        if self.config.show_fps:
            self.fps_monitor.update()
            self.renderer.draw_fps(result_frame, self.fps_monitor.get_fps())
        
        self.renderer.draw_instructions(result_frame)
        
        return result_frame
    
    def save_current_frame(self, frame: np.ndarray):
        """Save current frame to file"""
        timestamp = int(time.time())
        filename = f"smart_focus_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Frame saved as {filename}")
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting processing loop")
        logger.info("Click on objects to select them for tracking")
        logger.info("Press 'q' to quit, 's' to save frame")
        
        cv2.namedWindow('Smart Focus System')
        cv2.setMouseCallback('Smart Focus System', self.click_handler.mouse_callback)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Smart Focus System', processed_frame)
                
                # Save output
                if self.config.save_output and hasattr(self, 'out'):
                    self.out.write(processed_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_current_frame(processed_frame)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        if hasattr(self, 'out'):
            self.out.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['ultralytics', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("Installing missing packages...")
        
        import subprocess
        import sys
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        logger.info("Packages installed. Please restart the application.")
        sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI-Powered Smart Focus System")
    parser.add_argument('--input', type=str, default='0',
                       help='Input source: 0 for webcam or path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--save', action='store_true',
                       help='Save output video')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--blur', type=int, default=45,
                       help='Blur strength (odd number, default: 45)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--resize', type=int, default=640,
                       help='Resize width for faster processing (default: 640)')
    parser.add_argument('--no-flow', action='store_true',
                       help='Disable optical flow tracking')
    parser.add_argument('--no-kalman', action='store_true',
                       help='Disable Kalman filter')
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Create configuration
    config = Config(
        input_source=args.input,
        output_path=args.output,
        save_output=args.save,
        use_cuda=not args.no_cuda and torch.cuda.is_available(),
        blur_strength=args.blur,
        confidence_threshold=args.conf,
        resize_width=args.resize if args.resize > 0 else None,
        use_optical_flow=not args.no_flow,
        use_kalman=not args.no_kalman
    )
    
    # Create and run system
    try:
        system = SmartFocusSystem(config)
        system.run()
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()