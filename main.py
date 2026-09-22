"""Local dev entrypoint: the FastAPI app from app_core.py, plus the static
chat UI mounted at "/". Run with: uvicorn main:app --reload
"""

from fastapi.staticfiles import StaticFiles

from app_core import app

app.mount("/", StaticFiles(directory="static", html=True), name="static")
