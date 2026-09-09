@echo off
echo Starting S.A.F.E. FastAPI Backend and Streamlit Dashboard...
start "SAFE FastAPI Backend" cmd /k "python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
start "SAFE Streamlit Dashboard" cmd /k "python -m streamlit run src/dashboard/app.py --server.port 8501"
echo Services launched!
echo Dashboard: http://localhost:8501
echo API Docs:  http://localhost:8000/docs
pause
