@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt -q
python -m streamlit run investment_app/app.py
