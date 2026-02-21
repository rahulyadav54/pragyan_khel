from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64
import json
from app.services.ai_service import ai_service

router = APIRouter()

@router.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    
    try:
        while True:
            try:
                # Receive frame from client
                data = await websocket.receive_text()
                message = json.loads(data)
                message_type = message.get('type')

                if message_type == 'frame':
                    # Decode frame
                    frame_data = message.get('data')
                    if not frame_data:
                        continue

                    frame_bytes = base64.b64decode(frame_data.split(',', 1)[1] if ',' in frame_data else frame_data)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    blur_intensity = message.get('blur_intensity', 25)

                    # Process frame; if AI fails, return original frame to keep stream alive.
                    try:
                        processed_frame, detections = ai_service.process_frame(frame, blur_intensity)
                    except Exception as processing_error:
                        print(f"Frame processing error: {processing_error}")
                        processed_frame, detections = frame, []

                    # Encode result
                    ok, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    if not ok:
                        continue
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Send back
                    await websocket.send_json({
                        'type': 'frame',
                        'data': f'data:image/jpeg;base64,{frame_base64}',
                        'detections': detections
                    })

                elif message_type == 'select':
                    # Select object
                    x = message.get('x')
                    y = message.get('y')
                    if x is None or y is None:
                        continue

                    track_id = ai_service.select_object(x, y)

                    await websocket.send_json({
                        'type': 'selected',
                        'track_id': track_id
                    })

                elif message_type == 'reset':
                    # Reset tracking
                    ai_service.selected_track_id = None
                    ai_service.selected_box = None
                    ai_service.selected_cls = None
                    ai_service.selected_lost_frames = 0
                    ai_service.selected_point = None
                    ai_service.selected_point_velocity = np.array([0.0, 0.0], dtype=np.float32)
                    ai_service.point_lost_frames = 0
                    await websocket.send_json({
                        'type': 'reset',
                        'status': 'ok'
                    })
            except Exception as message_error:
                print(f"WebSocket message error: {message_error}")
                continue
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
