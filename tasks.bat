@echo off
IF "%1"=="setup"   ( py -3.11 -m venv .venv && call .\.venv\Scripts\activate && python -m pip install -U pip && pip install -r requirements.txt ) ^
ELSE IF "%1"=="download" ( call .\.venv\Scripts\activate && python -m src.data ) ^
ELSE IF "%1"=="train"   ( call .\.venv\Scripts\activate && python -m src.train ) ^
ELSE IF "%1"=="report"  ( call .\.venv\Scripts\activate && python -m src.report ) ^
ELSE IF "%1"=="app"     ( call .\.venv\Scripts\activate && streamlit run app\streamlit_app.py ) ^
ELSE ( echo Usage: tasks setup^|download^|train^|report^|app )
