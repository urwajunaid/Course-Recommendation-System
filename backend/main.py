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

    # Auto-seed courses table from CSV if it is empty
    try:
        from database import get_connection
        from ml.loader import get_courses_df
        import pandas as pd

        conn = get_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM courses")
                count = cursor.fetchone()["count"]

            if count == 0:
                print("[INFO] courses table is empty — seeding from CSV...")
                courses_df = get_courses_df()
                col_map = {
                    "Course_Name"        : "course_name",
                    "Company_Name"       : "company_name",
                    "Difficulty"         : "difficulty",
                    "Ratings"            : "ratings",
                    "Reviews"            : "reviews",
                    "Type_Of_Certificate": "type_of_certificate",
                    "Duration"           : "duration",
                    "Skills"             : "skills",
                }
                inserted = 0
                with conn.cursor() as cursor:
                    for _, row in courses_df.iterrows():
                        cols, vals = [], []
                        for csv_col, db_col in col_map.items():
                            if csv_col in row:
                                cols.append(db_col)
                                vals.append(None if pd.isna(row[csv_col]) else row[csv_col])
                        placeholders = ", ".join(["%s"] * len(vals))
                        col_names = ", ".join(cols)
                        cursor.execute(
                            f"INSERT INTO courses ({col_names}) VALUES ({placeholders})",
                            tuple(vals),
                        )
                        inserted += 1
                conn.commit()
                print(f"[OK] Seeded {inserted} courses into the database.")
            else:
                print(f"[OK] courses table already has {count} rows — skipping seed.")
            conn.close()
    except Exception as e:
        import traceback
        print(f"[WARN] Auto-seed failed: {e}")
        traceback.print_exc()

    yield


app = FastAPI(
    title="Course Recommendation API",
    description="Hybrid CBF + NCF recommendation engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",  # ← add yours
    "http://localhost:3000",
],
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
