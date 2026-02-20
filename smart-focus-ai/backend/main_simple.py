from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import uvicorn

app = FastAPI(title="Smart Focus AI - Quick Start")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple tracker
class SimpleTracker:
    def __init__(self):
        self.selected_box = None
        self.blur_intensity = 25
        
    def apply_blur(self, frame, box):
        if box is None:
            return frame
        
        x1, y1, x2, y2 = box
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Create elliptical mask
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = (x2 - x1) // 2
        height = (y2 - y1) // 2
        
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        # Apply blur
        blur_size = self.blur_intensity if self.blur_intensity % 2 == 1 else self.blur_intensity + 1
        blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        
        mask_norm = mask.astype(float) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        
        # Add glow
        cv2.rectangle(result, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 3)
        
        return result

tracker = SimpleTracker()

@app.get("/")
async def root():
    return {"message": "Smart Focus AI - Running", "status": "ready"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/api/reset-tracking")
async def reset():
    tracker.selected_box = None
    return {"status": "reset"}

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Decode frame
                frame_data = message['data']
                frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                tracker.blur_intensity = message.get('blur_intensity', 25)
                
                # Apply blur if object selected
                if tracker.selected_box:
                    frame = tracker.apply_blur(frame, tracker.selected_box)
                
                # Encode result
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    'type': 'frame',
                    'data': f'data:image/jpeg;base64,{frame_base64}',
                    'detections': []
                })
            
            elif message['type'] == 'select':
                x = message['x']
                y = message['y']
                
                # Create a box around clicked point
                box_size = 150
                h, w = 480, 640  # Approximate frame size
                x1 = max(0, x - box_size)
                y1 = max(0, y - box_size)
                x2 = min(w, x + box_size)
                y2 = min(h, y + box_size)
                
                tracker.selected_box = [x1, y1, x2, y2]
                
                await websocket.send_json({
                    'type': 'selected',
                    'track_id': 1
                })
            
            elif message['type'] == 'reset':
                tracker.selected_box = None
                await websocket.send_json({
                    'type': 'reset',
                    'status': 'ok'
                })
    
    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Smart Focus AI Backend...")
    print("📍 Server: http://localhost:8000")
    print("📍 Health: http://localhost:8000/health")
    print("✅ Ready to accept connections!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
