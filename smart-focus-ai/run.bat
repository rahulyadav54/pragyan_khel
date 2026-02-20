@echo off
echo Starting Smart Focus AI...
echo.

start "Backend" cmd /k "cd backend && venv\Scripts\activate && python main.py"
timeout /t 3 /nobreak > nul

start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Application starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
