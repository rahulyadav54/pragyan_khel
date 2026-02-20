@echo off
echo Installing ZCAM dependencies...
echo.

echo Installing Python packages...
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo You can now run the application with: python App.py
pause