@echo off
echo ========================================
echo Smart Focus AI - Setup Script
echo ========================================
echo.

echo [1/4] Setting up Backend...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
cd ..

echo.
echo [2/4] Setting up Frontend...
cd frontend
call npm install
cd ..

echo.
echo [3/4] Setup Complete!
echo.
echo ========================================
echo To run the application:
echo ========================================
echo.
echo Backend:  cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo Frontend: cd frontend ^&^& npm run dev
echo.
echo Then open: http://localhost:3000
echo ========================================
pause
