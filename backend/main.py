"""
============================================================
  Course Recommendation System — FastAPI Backend
============================================================
  Run:  cd backend && uvicorn main:app --reload --port 8000
============================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routers import auth, courses, recommend
from ml.loader import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to create DB tables (non-fatal if MySQL is unavailable)
    try:
        from database import create_tables
        create_tables()
        print("[OK] Database tables ready.")
    except Exception as e:
        print(f"[WARN] DB table creation failed: {e}")
        print("[WARN] Auth/rating features require a running MySQL instance.")

    # Load ML models once at startup
    load_models()
    yield


app = FastAPI(
    title="Course Recommendation API",
    description="Hybrid CBF + NCF recommendation engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api/auth",      tags=["Auth"])
app.include_router(courses.router,   prefix="/api/courses",   tags=["Courses"])
app.include_router(recommend.router, prefix="/api/recommend", tags=["Recommend"])


@app.get("/")
def root():
    return {"status": "ok", "message": "Course Recommender API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
