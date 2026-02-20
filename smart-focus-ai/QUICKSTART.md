# ⚡ Quick Start Guide - Smart Focus AI

## 🚀 Get Running in 5 Minutes

### Step 1: Setup (One-time)

**Windows Users:**
```bash
# Double-click or run:
setup.bat
```

**Mac/Linux Users:**
```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend (new terminal)
cd frontend
npm install
```

### Step 2: Run Application

**Windows:**
```bash
# Double-click or run:
run.bat
```

**Mac/Linux:**
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Step 3: Use the App

1. Open browser: **http://localhost:3000**
2. Upload a video or enable webcam
3. Click on any object to track it
4. Adjust blur intensity
5. Enjoy! 🎉

## 📋 Requirements

- Python 3.11+
- Node.js 18+
- 4GB RAM minimum
- Webcam (optional)

## 🎯 First Time Tips

1. **Start with upload mode** - Easier than webcam
2. **Click clearly on objects** - Center of object works best
3. **Adjust blur gradually** - Start at 25, then experiment
4. **Try different objects** - People, cars, animals all work!

## 🐛 Quick Troubleshooting

**Backend won't start?**
- Check Python version: `python --version` (need 3.11+)
- Install dependencies: `pip install -r requirements.txt`

**Frontend won't start?**
- Check Node version: `node --version` (need 18+)
- Delete `node_modules` and run `npm install` again

**Can't connect?**
- Make sure backend is running first
- Check http://localhost:8000/health

## 🎨 What You Can Do

✅ Track people in videos
✅ Track cars, animals, sports objects
✅ Create cinematic blur effects
✅ Process live webcam feed
✅ Switch between objects instantly
✅ Adjust blur intensity in real-time

## 📚 Learn More

- Full documentation: `README.md`
- Project details: `PROJECT_SUMMARY.md`
- API docs: http://localhost:8000/docs

---

**That's it! You're ready to go! 🚀**
