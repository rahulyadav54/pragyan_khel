import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectTracker:
    """Simple IoU-based object tracker with mask metadata."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 30

    def update(self, detections: List[dict]) -> dict:
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    "box": det["box"],
                    "mask": det.get("mask"),
                    "cls": det.get("cls"),
                    "conf": det.get("conf"),
                    "age": 0,
                }
                self.next_id += 1
            return self.tracks

        matched = set()
        for det in detections:
            det_box = det["box"]
            best_iou = 0
            best_id = None
            for tid, track in self.tracks.items():
                iou = self._compute_iou(det_box, track["box"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id] = {
                    "box": det_box,
                    "mask": det.get("mask"),
                    "cls": det.get("cls"),
                    "conf": det.get("conf"),
                    "age": 0,
                }
                matched.add(best_id)
            else:
                self.tracks[self.next_id] = {
                    "box": det_box,
                    "mask": det.get("mask"),
                    "cls": det.get("cls"),
                    "conf": det.get("conf"),
                    "age": 0,
                }
                matched.add(self.next_id)
                self.next_id += 1

        to_delete = []
        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
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
        self.selected_box = None
        self.selected_cls = None
        self.selected_lost_frames = 0
        self.max_selected_lost_frames = 20
        self.current_tracks = {}
        self.last_detections = []

        # Performance + robustness controls.
        self.frame_index = 0
        self.detection_interval = 3
        self.inference_size_idle = 384
        self.inference_size_active = 448
        self.last_results = None
        self.low_light_threshold = 88.0

        # Point-level tracking for moving targets (ex: pointed hand).
        self.prev_gray = None
        self.selected_point = None
        self.selected_point_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.point_lost_frames = 0

    async def initialize(self):
        """Initialize YOLO model (segmentation first)."""
        try:
            # Default to detection model for realtime FPS. Segmentation can be forced via MODEL_PATH.
            preferred_model = os.getenv("MODEL_PATH", "yolo11n.pt")
            try:
                self.detector = YOLO(preferred_model)
                print(f"YOLO model loaded: {preferred_model}")
            except Exception:
                fallback_model = "yolov8n.pt"
                self.detector = YOLO(fallback_model)
                print(f"YOLO model loaded: {fallback_model}")
            self.is_initialized = True
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            self.is_initialized = False

    def detect_objects(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect objects and optional segmentation masks in frame."""
        if not self.is_initialized:
            return [], frame

        h, w = frame.shape[:2]

        # Apply enhancement only when needed to avoid extra per-frame cost.
        try:
            if float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < self.low_light_threshold:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

        scale = 1.0
        max_side = max(h, w)
        inference_size = self.inference_size_active if self.selected_track_id is not None else self.inference_size_idle

        if max_side > inference_size:
            scale = inference_size / float(max_side)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        results = self.detector(resized, imgsz=inference_size, verbose=False)[0]
        detections = []
        masks_data = getattr(results, "masks", None)
        masks_tensor = masks_data.data if masks_data is not None else None

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if scale != 1.0:
                inv = 1.0 / scale
                x1 = int(x1 * inv)
                y1 = int(y1 * inv)
                x2 = int(x2 * inv)
                y2 = int(y2 * inv)

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf < 0.28:
                continue

            mask = None
            if masks_tensor is not None and i < len(masks_tensor):
                raw_mask = masks_tensor[i].cpu().numpy()
                mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8) * 255

            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": conf,
                "cls": cls,
                "mask": mask,
            })

        return detections, results

    def track_objects(self, detections: List[dict]) -> dict:
        """Update tracker with new detections."""
        self.current_tracks = self.tracker.update(detections)
        return self.current_tracks

    def select_object(self, x: int, y: int) -> Optional[int]:
        """Select object at position."""
        best_tid = None
        smallest_area = float("inf")

        for tid, track in self.current_tracks.items():
            box = track["box"]
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
                if area < smallest_area:
                    smallest_area = area
                    best_tid = tid

        if best_tid is None:
            return None

        self.selected_track_id = best_tid
        self.selected_box = self.current_tracks[best_tid]["box"]
        self.selected_cls = self.current_tracks[best_tid].get("cls")
        self.selected_lost_frames = 0
        self.selected_point = np.array([float(x), float(y)], dtype=np.float32)
        self.selected_point_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.point_lost_frames = 0
        return best_tid

    def _box_from_point(self, point: np.ndarray, box: List[int], frame_shape: Tuple[int, int, int]) -> List[int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bw = max(8, x2 - x1)
        bh = max(8, y2 - y1)
        cx = int(np.clip(point[0], 0, w - 1))
        cy = int(np.clip(point[1], 0, h - 1))
        nx1 = max(0, min(w - 1, cx - bw // 2))
        ny1 = max(0, min(h - 1, cy - bh // 2))
        nx2 = max(nx1 + 1, min(w, nx1 + bw))
        ny2 = max(ny1 + 1, min(h, ny1 + bh))
        return [nx1, ny1, nx2, ny2]

    def _update_selected_point(self, frame: np.ndarray, selected_track: Optional[dict]) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.selected_point is not None and self.prev_gray is not None:
            p0 = self.selected_point.reshape(1, 1, 2).astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                p0,
                None,
                winSize=(19, 19),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03),
            )
            if p1 is not None and st is not None and int(st[0][0]) == 1:
                new_point = p1[0][0]
                self.selected_point_velocity = (new_point - self.selected_point) * 0.6 + (self.selected_point_velocity * 0.4)
                self.selected_point = new_point.astype(np.float32)
                self.point_lost_frames = 0
            else:
                self.point_lost_frames += 1
                self.selected_point = (self.selected_point + self.selected_point_velocity).astype(np.float32)
                self.selected_point_velocity *= 0.85

        if self.selected_point is not None and selected_track is not None:
            bx1, by1, bx2, by2 = selected_track["box"]
            self.selected_point[0] = float(np.clip(self.selected_point[0], bx1, max(bx1, bx2 - 1)))
            self.selected_point[1] = float(np.clip(self.selected_point[1], by1, max(by1, by2 - 1)))

        self.prev_gray = gray

    def _recover_selected_track(self, tracks: dict) -> None:
        if self.selected_track_id in tracks or self.selected_box is None:
            return

        best_tid = None
        best_score = 0.0
        sb = self.selected_box
        sb_center = ((sb[0] + sb[2]) / 2.0, (sb[1] + sb[3]) / 2.0)
        sb_diag = max(1.0, np.hypot(sb[2] - sb[0], sb[3] - sb[1]))

        for tid, track in tracks.items():
            track_box = track["box"]
            iou = self.tracker._compute_iou(sb, track_box)
            tcx = (track_box[0] + track_box[2]) / 2.0
            tcy = (track_box[1] + track_box[3]) / 2.0
            center_dist = np.hypot(tcx - sb_center[0], tcy - sb_center[1]) / sb_diag
            class_bonus = 0.08 if self.selected_cls is not None and track.get("cls") == self.selected_cls else 0.0
            score = (iou * 0.65) + (max(0.0, 1.0 - center_dist) * 0.27) + class_bonus
            if score > best_score:
                best_score = score
                best_tid = tid

        if best_tid is not None and best_score >= 0.18:
            self.selected_track_id = best_tid
            self.selected_box = tracks[best_tid]["box"]
            self.selected_cls = tracks[best_tid].get("cls")
            self.selected_lost_frames = 0
        else:
            self.selected_lost_frames += 1
            if self.selected_lost_frames > self.max_selected_lost_frames:
                self.selected_track_id = None
                self.selected_box = None
                self.selected_cls = None
                self.selected_point = None
                self.selected_point_velocity = np.array([0.0, 0.0], dtype=np.float32)
                self.point_lost_frames = 0
                self.selected_lost_frames = 0

    def create_mask(self, frame: np.ndarray, track: dict) -> np.ndarray:
        """Create focus mask using segmentation, fallback to soft rectangular subject mask."""
        seg_mask = track.get("mask")
        if seg_mask is not None and np.any(seg_mask):
            kernel = np.ones((5, 5), np.uint8)
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            seg_mask = cv2.GaussianBlur(seg_mask, (9, 9), 0)
            base_mask = seg_mask
        else:
            fallback = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, track["box"])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                fallback[y1:y2, x1:x2] = 255
                edge = max(5, (min(x2 - x1, y2 - y1) // 8) | 1)
                fallback = cv2.GaussianBlur(fallback, (edge, edge), 0)
            base_mask = fallback

        # Point-priority focus: keep clicked moving point (ex: hand) sharp while still using subject mask.
        if self.selected_point is not None:
            px, py = int(self.selected_point[0]), int(self.selected_point[1])
            x1, y1, x2, y2 = map(int, track["box"])
            diag = max(20.0, float(np.hypot(x2 - x1, y2 - y1)))
            radius = int(np.clip(diag * 0.16, 22, 90))
            point_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(point_mask, (px, py), radius, 255, -1)
            point_mask = cv2.GaussianBlur(point_mask, (19, 19), 0)
            base_mask = np.maximum(base_mask, point_mask)

        return base_mask

    def _set_adaptive_runtime(self) -> None:
        if self.selected_track_id is None:
            self.detection_interval = 4
            return
        speed = float(np.linalg.norm(self.selected_point_velocity))
        if speed > 11.0:
            self.detection_interval = 1
        elif speed > 5.0:
            self.detection_interval = 2
        else:
            self.detection_interval = 3

    def apply_blur(self, frame: np.ndarray, mask: np.ndarray, blur_intensity: int = 25) -> np.ndarray:
        """Apply background blur."""
        blur_intensity = max(5, min(99, int(blur_intensity)))
        if blur_intensity % 2 == 0:
            blur_intensity += 1

        # Faster cinematic blur: blur in lower-res space, then upscale.
        h, w = frame.shape[:2]
        scale = 0.5 if max(h, w) >= 720 else 0.65
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        small_blur_k = max(5, int((blur_intensity * scale) // 2 * 2 + 1))
        blurred_small = cv2.GaussianBlur(small, (small_blur_k, small_blur_k), 0)
        blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_norm = mask.astype(float) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)

        return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    def process_frame(self, frame: np.ndarray, blur_intensity: int = 25) -> Tuple[np.ndarray, List]:
        """Process single frame."""
        self.frame_index += 1
        self._set_adaptive_runtime()
        should_detect = (self.frame_index % self.detection_interval == 0) or (not self.current_tracks)

        if should_detect:
            detections, results = self.detect_objects(frame)
            tracks = self.track_objects(detections)
            self.last_results = results
            self.last_detections = detections
        else:
            tracks = self.current_tracks

        self._recover_selected_track(tracks)

        processed_frame = frame.copy()
        detection_data = []

        if self.selected_track_id is not None and self.selected_track_id in tracks:
            tracked = tracks[self.selected_track_id]
            self._update_selected_point(frame, tracked)
            if self.selected_point is not None:
                tracked["box"] = self._box_from_point(self.selected_point, tracked["box"], frame.shape)
            mask = self.create_mask(frame, tracked)
            processed_frame = self.apply_blur(frame, mask, blur_intensity)
            self.selected_box = tracked["box"]
            self.selected_cls = tracked.get("cls")
            self.selected_lost_frames = 0
        else:
            self._update_selected_point(frame, None)

        for tid, track in tracks.items():
            box = track["box"]
            detection_data.append({
                "id": tid,
                "box": box,
                "selected": tid == self.selected_track_id,
            })

        return processed_frame, detection_data


# Global AI service instance
ai_service = AIService()
