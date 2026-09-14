"""
uvicorn 진입점. 실제 FastAPI 앱 구현은 src/api/main.py에 있음.

실행: .\venv_blackwell\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8082
"""

from src.api.main import app
