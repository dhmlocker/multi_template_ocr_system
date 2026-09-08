@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  echo [1/4] Creating .venv ...
  py -3 -m venv .venv
  if errorlevel 1 python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if not exist requirements.lock.txt (
  echo requirements.lock.txt not found.
  exit /b 1
)
echo [2/4] Installing the verified runtime ...
pip install -r requirements.lock.txt
if errorlevel 1 (
  echo Locked package installation failed. Check Python version and PaddlePaddle wheel availability.
  exit /b 1
)
echo [3/4] Downloading/checking official PaddleOCRv6 models ...
python tools\download_models.py --device cpu
if errorlevel 1 exit /b 1
echo [4/4] Runtime manifest:
python tools\print_runtime_manifest.py

echo Starting Streamlit at http://localhost:8501 ...
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
