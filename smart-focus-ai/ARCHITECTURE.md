# 🏗️ Smart Focus AI - Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                             │
│                     http://localhost:3000                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/WebSocket
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    REACT FRONTEND (Vite)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  App.jsx (Main Component)                                 │  │
│  │  ├── VideoPlayer.jsx (Video Upload & Processing)         │  │
│  │  ├── WebcamCapture.jsx (Live Webcam Feed)                │  │
│  │  └── Controls.jsx (UI Controls & Settings)               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Features:                                                        │
│  • Tailwind CSS (Styling)                                        │
│  • Framer Motion (Animations)                                    │
│  • WebSocket Client (Real-time streaming)                        │
│  • Axios (HTTP requests)                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ WebSocket: /ws/video
                         │ REST API: /api/*
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   FASTAPI BACKEND (Python)                       │
│                    http://localhost:8000                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  main.py (FastAPI App)                                    │  │
│  │  ├── CORS Middleware                                      │  │
│  │  ├── Routes                                               │  │
│  │  └── WebSocket Handler                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────────┐  │
│  │  Routes Layer                                             │  │
│  │  ├── video_routes.py (REST API)                          │  │
│  │  │   ├── POST /api/upload-video                          │  │
│  │  │   ├── POST /api/process-frame                         │  │
│  │  │   ├── POST /api/select-object                         │  │
│  │  │   └── POST /api/reset-tracking                        │  │
│  │  │                                                         │  │
│  │  └── websocket_routes.py (WebSocket)                     │  │
│  │      └── WS /ws/video (Real-time streaming)              │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────────┐  │
│  │  Services Layer (Business Logic)                         │  │
│  │  └── ai_service.py                                        │  │
│  │      ├── AIService (Main service)                        │  │
│  │      ├── ObjectTracker (IoU tracking)                    │  │
│  │      └── Processing Pipeline                             │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────────┐  │
│  │  AI Models & Processing                                  │  │
│  │  ├── YOLOv8-seg (Detection + Segmentation)               │  │
│  │  ├── IoU Tracker (Object tracking)                       │  │
│  │  ├── Mask Generator (Segmentation masks)                 │  │
│  │  └── Blur Engine (Background blur)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Video Upload Flow
```
User → Upload Video → Frontend
                        ↓
                   Validate File
                        ↓
                   POST /api/upload-video → Backend
                                              ↓
                                         Save Temp File
                                              ↓
                                         Extract Metadata
                                              ↓
                                         Return Info → Frontend
```

### 2. Real-Time Processing Flow (WebSocket)
```
Frontend                          Backend
   │                                 │
   ├─ Connect WebSocket ────────────→ Accept Connection
   │                                 │
   ├─ Send Frame (base64) ──────────→ Receive Frame
   │                                 │
   │                                 ├─ Decode Frame
   │                                 │
   │                                 ├─ Run YOLO Detection
   │                                 │
   │                                 ├─ Update Tracker
   │                                 │
   │                                 ├─ Generate Mask (if selected)
   │                                 │
   │                                 ├─ Apply Blur
   │                                 │
   │                                 ├─ Encode Result
   │                                 │
   │←─ Receive Processed Frame ──────┤ Send Frame + Detections
   │                                 │
   └─ Display on Canvas              │
```

### 3. Object Selection Flow
```
User Click → Get Coordinates → Frontend
                                  ↓
                            Send via WebSocket
                                  ↓
                              Backend
                                  ↓
                         Check All Tracked Objects
                                  ↓
                         Find Object at Position
                                  ↓
                         Set as Selected Track
                                  ↓
                         Return Track ID → Frontend
                                              ↓
                                         Update UI
```

## Component Breakdown

### Frontend Components

#### App.jsx
- **Purpose**: Main application container
- **State Management**:
  - mode (upload/webcam)
  - videoFile
  - blurIntensity
  - selectedObject
  - fps
- **Responsibilities**:
  - Route between upload/webcam modes
  - Manage global state
  - Handle file uploads
  - Coordinate child components

#### VideoPlayer.jsx
- **Purpose**: Handle uploaded video processing
- **Key Features**:
  - WebSocket connection
  - Frame extraction from video
  - Send frames to backend
  - Display processed frames
  - Handle object selection clicks
  - FPS calculation

#### WebcamCapture.jsx
- **Purpose**: Handle live webcam feed
- **Key Features**:
  - Access user webcam
  - Capture frames in real-time
  - Send to backend via WebSocket
  - Display processed stream
  - Handle object selection

#### Controls.jsx
- **Purpose**: UI controls and settings
- **Features**:
  - Blur intensity slider
  - Reset button
  - Status indicators
  - Stats display

### Backend Services

#### AIService
- **Purpose**: Core AI processing logic
- **Methods**:
  - `initialize()` - Load YOLO model
  - `detect_objects()` - Run object detection
  - `track_objects()` - Update tracker
  - `select_object()` - Select object at position
  - `create_mask()` - Generate segmentation mask
  - `apply_blur()` - Apply background blur
  - `process_frame()` - Main processing pipeline

#### ObjectTracker
- **Purpose**: Track objects across frames
- **Algorithm**: IoU (Intersection over Union)
- **Features**:
  - Assign unique IDs
  - Match detections across frames
  - Handle occlusion (age tracking)
  - Remove lost tracks

## Processing Pipeline

```
Input Frame
    ↓
┌───────────────────┐
│ YOLO Detection    │ → Detect 80+ object classes
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Extract Boxes     │ → Get bounding boxes
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Update Tracker    │ → Assign/update track IDs
└────────┬──────────┘
         ↓
    Is Object Selected?
         ↓
    ┌────┴────┐
   Yes        No
    ↓          ↓
┌───────────────────┐    ┌──────────────┐
│ Get Selected Box  │    │ Return Frame │
└────────┬──────────┘    └──────────────┘
         ↓
┌───────────────────┐
│ Generate Mask     │ → Segmentation or ellipse
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Apply Blur        │ → Gaussian blur on background
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Add Glow Effect   │ → Highlight selected object
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Encode Frame      │ → Convert to JPEG base64
└────────┬──────────┘
         ↓
    Output Frame
```

## API Endpoints

### REST API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/api/upload-video` | Upload video file |
| POST | `/api/process-frame` | Process single frame |
| POST | `/api/select-object` | Select object at position |
| POST | `/api/reset-tracking` | Reset tracking state |

### WebSocket

| Endpoint | Direction | Message Type | Purpose |
|----------|-----------|--------------|---------|
| `/ws/video` | Client → Server | `frame` | Send frame for processing |
| `/ws/video` | Server → Client | `frame` | Return processed frame |
| `/ws/video` | Client → Server | `select` | Select object |
| `/ws/video` | Server → Client | `selected` | Confirm selection |
| `/ws/video` | Client → Server | `reset` | Reset tracking |
| `/ws/video` | Server → Client | `reset` | Confirm reset |

## Performance Optimization

### Backend Optimizations
1. **Model Loading**: Load once on startup
2. **Frame Processing**: Async processing
3. **Image Encoding**: JPEG with quality 85
4. **Tracking**: Efficient IoU calculation
5. **Memory**: Reuse arrays where possible

### Frontend Optimizations
1. **Frame Rate**: Limit to 30 FPS
2. **Canvas Rendering**: Direct canvas manipulation
3. **WebSocket**: Binary data transfer
4. **State Management**: Minimal re-renders
5. **Animations**: GPU-accelerated (Framer Motion)

## Security Considerations

1. **CORS**: Configured for development (restrict in production)
2. **File Upload**: Validate file types and sizes
3. **WebSocket**: Connection limits (implement in production)
4. **Input Validation**: Validate all user inputs
5. **Error Handling**: Don't expose internal errors

## Scalability

### Current Limitations
- Single-threaded processing
- In-memory state
- No persistent storage

### Scaling Strategies
1. **Horizontal Scaling**: Multiple backend instances
2. **Load Balancing**: Nginx or cloud LB
3. **GPU Acceleration**: CUDA support
4. **Caching**: Redis for session state
5. **Queue System**: RabbitMQ for frame processing
6. **CDN**: Serve frontend from CDN

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Production Setup                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Vercel     │         │   Render     │             │
│  │  (Frontend)  │────────→│  (Backend)   │             │
│  │              │  HTTPS  │              │             │
│  └──────────────┘         └──────────────┘             │
│         │                        │                       │
│         │                        │                       │
│         ↓                        ↓                       │
│  ┌──────────────┐         ┌──────────────┐             │
│  │     CDN      │         │  GPU Server  │             │
│  │  (Static)    │         │  (Optional)  │             │
│  └──────────────┘         └──────────────┘             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Technology Decisions

### Why FastAPI?
- Modern Python framework
- Built-in WebSocket support
- Auto-generated API docs
- High performance (async)
- Easy to deploy

### Why React + Vite?
- Fast development
- Modern tooling
- Great ecosystem
- Easy to learn
- Production-ready

### Why YOLOv8?
- State-of-the-art accuracy
- Real-time performance
- Segmentation support
- Easy to use (Ultralytics)
- Active development

### Why Tailwind CSS?
- Utility-first approach
- Fast development
- Small bundle size
- Consistent design
- Easy customization

## Future Enhancements

1. **Authentication**: User accounts
2. **Storage**: Save processed videos
3. **Multi-object**: Track multiple objects
4. **Depth**: Real depth estimation
5. **Mobile**: React Native app
6. **Analytics**: Usage tracking
7. **API**: Public API with rate limiting
8. **Collaboration**: Real-time multi-user

---

**This architecture is designed for:**
- ✅ Easy development
- ✅ Fast performance
- ✅ Simple deployment
- ✅ Future scalability
- ✅ Maintainability
