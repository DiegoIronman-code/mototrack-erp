@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Preparando el entorno por primera vez, esto puede tardar unos minutos...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo Iniciando la aplicacion...
streamlit run app.py

pause
