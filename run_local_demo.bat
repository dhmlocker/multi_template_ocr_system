@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python tools\check_local_env.py
if errorlevel 1 exit /b 1
python tools\run_demo_samples.py --only 医院收费票据
if errorlevel 1 exit /b 1
python tools\run_demo_samples.py --only 增值稅发票
if errorlevel 1 exit /b 1
python tools\run_demo_samples.py --only 快递寄件面单
if errorlevel 1 exit /b 1
python tools\make_contact_sheet.py
python tools\print_runtime_manifest.py > outputs\local_runtime_manifest.json
if errorlevel 1 exit /b 1
echo Local demo completed. Open outputs\demo_samples for OCR box images, white result images, official JSON and official PNG outputs.
