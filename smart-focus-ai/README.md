# 🎯 Smart Focus AI

A production-ready, full-stack AI-powered web application for real-time object tracking and cinematic background blur. Built with React, FastAPI, and YOLOv8.

![Smart Focus AI](https://img.shields.io/badge/AI-Powered-blue) ![React](https://img.shields.io/badge/React-18.2-61DAFB) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-FF6F00)

## ✨ Features

- 🎥 **Upload Video or Use Webcam** - Support for both video files and live camera feed
- 🎯 **Click-to-Track** - Simply click any object to track it across frames
- 🌫️ **Smart Background Blur** - AI-powered cinematic blur with adjustable intensity
- 🔄 **Real-time Processing** - 24-30 FPS performance with WebSocket streaming
- 🎨 **Modern UI** - Glassmorphism design with Framer Motion animations
- 📱 **Fully Responsive** - Works on desktop, tablet, and mobile
- 🚀 **Production Ready** - Clean architecture, error handling, and optimization

## 🏗️ Architecture

```
smart-focus-ai/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── services/       # AI Service (YOLO, Tracking, Blur)
│   │   ├── routes/         # API Routes & WebSocket
│   │   ├── models/         # Data Models
│   │   └── utils/          # Helper Functions
│   ├── main.py             # FastAPI App Entry
│   └── requirements.txt    # Python Dependencies
│
└── frontend/               # React Frontend
    ├── src/
    │   ├── components/     # React Components
    │   ├── utils/          # Utilities
    │   ├── App.jsx         # Main App
    │   └── main.jsx        # Entry Point
    ├── package.json        # Node Dependencies
    └── vite.config.js      # Vite Configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (automatic on first run)
# Or manually: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt

# Run backend
python main.py
```

Backend will start on `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will start on `http://localhost:3000`

## 📖 Usage

1. **Open the app** at `http://localhost:3000`
2. **Choose mode**: Upload video or enable webcam
3. **Click on any object** in the video to track it
4. **Adjust blur intensity** using the slider
5. **Click another object** to switch tracking
6. **Reset** to clear tracking

## 🔌 API Endpoints

### REST API

- `GET /` - API status
- `GET /health` - Health check
- `POST /api/upload-video` - Upload video file
- `POST /api/process-frame` - Process single frame
- `POST /api/select-object` - Select object at position
- `POST /api/reset-tracking` - Reset tracking state

### WebSocket

- `WS /ws/video` - Real-time video streaming
  - Send: `{ type: 'frame', data: base64, blur_intensity: 25 }`
  - Receive: `{ type: 'frame', data: base64, detections: [...] }`

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **YOLOv8-seg** - Object detection & segmentation
- **OpenCV** - Computer vision operations
- **NumPy** - Numerical computing
- **WebSocket** - Real-time communication

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Lucide React** - Icons

## 🎨 Features in Detail

### 1. Object Detection
- Detects 80+ object classes (COCO dataset)
- Real-time detection at 30 FPS
- Confidence threshold filtering

### 2. Object Tracking
- IoU-based tracking algorithm
- Maintains object identity across frames
- Handles occlusion and re-identification

### 3. Segmentation & Blur
- Pixel-perfect segmentation masks
- Gaussian blur with adjustable intensity
- Smooth edge blending
- Cinematic bokeh effect

### 4. User Interface
- Glassmorphism design
- Smooth animations
- Responsive layout
- Real-time FPS counter
- Loading states & error handling

## 📊 Performance

- **FPS**: 24-30 frames per second
- **Latency**: <50ms processing time
- **Memory**: ~500MB with YOLO model loaded
- **Supported Resolutions**: Up to 1920x1080

## 🚢 Deployment

### Deploy Backend (Render)

1. Create account on [Render](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### Deploy Frontend (Vercel)

```bash
cd frontend
npm run build
npx vercel --prod
```

Or connect GitHub repo to Vercel for automatic deployments.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## 🔧 Configuration

### Backend (.env)
```env
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=*
MODEL_PATH=yolov8n-seg.pt
```

### Frontend
Update API URL in `vite.config.js` for production:
```js
proxy: {
  '/api': {
    target: 'https://your-backend-url.com',
    changeOrigin: true
  }
}
```

## 🐛 Troubleshooting

### Backend Issues
- **YOLO model not found**: Model downloads automatically on first run
- **Port already in use**: Change port in `main.py`
- **CUDA errors**: CPU mode is used by default

### Frontend Issues
- **WebSocket connection failed**: Ensure backend is running
- **Webcam not working**: Check browser permissions
- **Slow performance**: Reduce video resolution or blur intensity

## 📝 License

MIT License - feel free to use for personal or commercial projects

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Support

For issues or questions, open a GitHub issue or contact support.

---

**Built with ❤️ using React, FastAPI, and YOLOv8**
