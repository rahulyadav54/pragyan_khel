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
            # Receive frame from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Decode frame
                frame_data = message['data']
                frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                blur_intensity = message.get('blur_intensity', 25)
                
                # Process frame
                processed_frame, detections = ai_service.process_frame(frame, blur_intensity)
                
                # Encode result
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send back
                await websocket.send_json({
                    'type': 'frame',
                    'data': f'data:image/jpeg;base64,{frame_base64}',
                    'detections': detections
                })
            
            elif message['type'] == 'select':
                # Select object
                x = message['x']
                y = message['y']
                track_id = ai_service.select_object(x, y)
                
                await websocket.send_json({
                    'type': 'selected',
                    'track_id': track_id
                })
            
            elif message['type'] == 'reset':
                # Reset tracking
                ai_service.selected_track_id = None
                await websocket.send_json({
                    'type': 'reset',
                    'status': 'ok'
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
