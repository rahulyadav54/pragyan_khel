import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List
import asyncio

class ObjectTracker:
    """Simple IoU-based object tracker"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 30
        
    def update(self, detections: List[List[int]]) -> dict:
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {'box': det, 'age': 0}
                self.next_id += 1
            return self.tracks
        
        matched = set()
        for det in detections:
            best_iou = 0
            best_id = None
            for tid, track in self.tracks.items():
                iou = self._compute_iou(det, track['box'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = tid
            
            if best_id:
                self.tracks[best_id] = {'box': det, 'age': 0}
                matched.add(best_id)
            else:
                self.tracks[self.next_id] = {'box': det, 'age': 0}
                self.next_id += 1
        
        to_delete = []
        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]
        
        return self.tracks
    
    def _compute_iou(self, box1, box2):
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

class AIService:
    def __init__(self):
        self.detector = None
        self.tracker = ObjectTracker()
        self.is_initialized = False
        self.selected_track_id = None
        self.current_tracks = {}
        
    async def initialize(self):
        """Initialize YOLO model"""
        try:
            self.detector = YOLO('yolov8n-seg.pt')
            self.is_initialized = True
            print("✅ YOLO segmentation model loaded")
        except:
            try:
                self.detector = YOLO('yolov8n.pt')
                self.is_initialized = True
                print("✅ YOLO detection model loaded")
            except Exception as e:
                print(f"❌ Failed to load YOLO: {e}")
                self.is_initialized = False
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect objects in frame"""
        if not self.is_initialized:
            return [], frame
        
        results = self.detector(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls])
        
        return detections, results
    
    def track_objects(self, detections: List) -> dict:
        """Update tracker with new detections"""
        boxes = [[d[0], d[1], d[2], d[3]] for d in detections]
        self.current_tracks = self.tracker.update(boxes)
        return self.current_tracks
    
    def select_object(self, x: int, y: int) -> Optional[int]:
        """Select object at position"""
        for tid, track in self.current_tracks.items():
            box = track['box']
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                self.selected_track_id = tid
                return tid
        return None
    
    def create_mask(self, frame: np.ndarray, box: List[int], results=None) -> np.ndarray:
        """Create segmentation mask"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = map(int, box)
        
        # Try to use YOLO segmentation mask if available
        if hasattr(results, 'masks') and results.masks is not None:
            try:
                for i, result_box in enumerate(results.boxes):
                    result_coords = list(map(int, result_box.xyxy[0]))
                    if result_coords == [x1, y1, x2, y2]:
                        seg_mask = results.masks.data[i].cpu().numpy()
                        mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                        mask = (mask * 255).astype(np.uint8)
                        return mask
            except:
                pass
        
        # Fallback: elliptical mask
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = (x2 - x1) // 2
        height = (y2 - y1) // 2
        
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        return mask
    
    def apply_blur(self, frame: np.ndarray, mask: np.ndarray, blur_intensity: int = 25) -> np.ndarray:
        """Apply background blur"""
        if blur_intensity % 2 == 0:
            blur_intensity += 1
        
        blurred = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)
        mask_norm = mask.astype(float) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        return result
    
    def process_frame(self, frame: np.ndarray, blur_intensity: int = 25) -> Tuple[np.ndarray, List]:
        """Process single frame"""
        detections, results = self.detect_objects(frame)
        tracks = self.track_objects(detections)
        
        processed_frame = frame.copy()
        detection_data = []
        
        # Apply blur if object is selected
        if self.selected_track_id and self.selected_track_id in tracks:
            tracked_box = tracks[self.selected_track_id]['box']
            mask = self.create_mask(frame, tracked_box, results)
            processed_frame = self.apply_blur(frame, mask, blur_intensity)
            
            # Add glow effect to selected object
            x1, y1, x2, y2 = tracked_box
            cv2.rectangle(processed_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 3)
        
        # Prepare detection data
        for tid, track in tracks.items():
            box = track['box']
            detection_data.append({
                'id': tid,
                'box': box,
                'selected': tid == self.selected_track_id
            })
        
        return processed_frame, detection_data

# Global AI service instance
ai_service = AIService()
