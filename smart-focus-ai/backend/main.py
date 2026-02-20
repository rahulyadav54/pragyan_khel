from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.routes import video_routes, websocket_routes
from app.services.ai_service import ai_service
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Smart Focus AI",
    description="AI-powered object tracking and background blur",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(video_routes.router, prefix="/api", tags=["video"])
app.include_router(websocket_routes.router, tags=["websocket"])

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await ai_service.initialize()
    print("✅ Smart Focus AI Backend Started")

@app.get("/")
async def root():
    return {"message": "Smart Focus AI API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": ai_service.is_initialized}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
