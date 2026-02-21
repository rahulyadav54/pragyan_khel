from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from app.services.ai_service import ai_service
import io
from PIL import Image

router = APIRouter()

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and validate video file"""
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    contents = await file.read()
    
    # Save temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    
    # Validate video
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "filename": file.filename,
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "path": temp_path
    }

@router.post("/process-frame")
async def process_frame(frame_data: dict):
    """Process single frame with AI"""
    # Decode base64 frame
    import base64
    frame_bytes = base64.b64decode(frame_data['frame'])
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    blur_intensity = frame_data.get('blur_intensity', 25)
    
    # Process frame
    processed_frame, detections = ai_service.process_frame(frame, blur_intensity)
    
    # Encode result
    _, buffer = cv2.imencode('.jpg', processed_frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "frame": frame_base64,
        "detections": detections
    }

@router.post("/select-object")
async def select_object(data: dict):
    """Select object at position"""
    x = data.get('x')
    y = data.get('y')
    
    if x is None or y is None:
        raise HTTPException(status_code=400, detail="Missing coordinates")
    
    track_id = ai_service.select_object(x, y)
    
    return {
        "selected": track_id is not None,
        "track_id": track_id
    }

@router.post("/reset-tracking")
async def reset_tracking():
    """Reset tracking state"""
    ai_service.selected_track_id = None
    ai_service.selected_box = None
    ai_service.selected_cls = None
    ai_service.selected_lost_frames = 0
    ai_service.selected_point = None
    ai_service.selected_point_velocity = np.array([0.0, 0.0], dtype=np.float32)
    ai_service.point_lost_frames = 0
    ai_service.tracker = ai_service.tracker.__class__()
    ai_service.current_tracks = {}
    
    return {"status": "reset"}
