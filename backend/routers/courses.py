"""
routers/courses.py — List courses, search, rate a course, seed DB from CSV
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import os, pandas as pd

from database import get_db, User
from security import get_current_user, get_optional_user
from ml.loader import get_courses_df

router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────

class RatingRequest(BaseModel):
    course_name : str
    rating      : float = Field(..., ge=1.0, le=5.0)


class CourseOut(BaseModel):
    id                  : int
    course_name         : str
    company_name        : Optional[str]
    difficulty          : Optional[str]
    ratings             : Optional[float]
    reviews             : Optional[str]
    type_of_certificate : Optional[str]
    duration            : Optional[str]

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────

@router.get("/test")
def test_endpoint():
    """Test endpoint to verify router is working."""
    return {"status": "courses router is working"}

@router.get("/", response_model=list[CourseOut])
def list_courses(
    page  : int = Query(1, ge=1),
    limit : int = Query(20, ge=1, le=100),
    db    = Depends(get_db),
):
    if not db:
        return []
    offset = (page - 1) * limit
    with db.cursor() as cursor:
        cursor.execute("SELECT * FROM courses LIMIT %s OFFSET %s", (limit, offset))
        rows = cursor.fetchall()
    return rows


@router.get("/search")
def search_courses(q: str = Query(..., min_length=1), db = Depends(get_db)):
    """Full-text substring search on course names."""
    if not db:
        return []
    with db.cursor() as cursor:
        cursor.execute("SELECT id, course_name, company_name, difficulty FROM courses WHERE course_name LIKE %s LIMIT 20", (f"%{q}%",))
        rows = cursor.fetchall()
    return rows


@router.get("/autocomplete")
def autocomplete(q: str = Query(..., min_length=1)):
    """Fast autocomplete from in-memory DataFrame (no DB hit)."""
    courses_df = get_courses_df()
    if courses_df is None:
        return []
    mask    = courses_df["Course_Name"].str.lower().str.contains(q.lower(), na=False)
    results = courses_df[mask]["Course_Name"].head(10).tolist()
    return results


@router.post("/rate", status_code=201)
def rate_course(
    body: RatingRequest,
    current_user: User = Depends(get_optional_user),
    db=Depends(get_db),
):
    """Save or update a user's rating for a course."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in first.")

    with db.cursor() as cursor:
        # ---------- FIND COURSE ----------
        cursor.execute(
            "SELECT id FROM courses WHERE LOWER(course_name) = LOWER(%s)",
            (body.course_name.strip(),)
        )
        course = cursor.fetchone()

        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        course_id = course["id"]

        # ---------- UPSERT RATING (atomic, prevents race conditions) ----------
        cursor.execute(
            """
            INSERT INTO interactions (user_id, course_id, rating)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE rating = VALUES(rating)
            """,
            (current_user.id, course_id, body.rating)
        )

    db.commit()
    return {"message": "Rating saved", "rating": body.rating}

@router.get("/my-ratings")
def my_ratings(current_user: User = Depends(get_current_user),
               db = Depends(get_db)):
    if not db:
        return []
    with db.cursor() as cursor:
        cursor.execute("""
            SELECT i.course_id, c.course_name, i.rating, i.created_at as rated_at
            FROM interactions i
            JOIN courses c ON i.course_id = c.id
            WHERE i.user_id = %s
        """, (current_user.id,))
        rows = cursor.fetchall()
    return rows


@router.post("/seed", status_code=201, tags=["Admin"])
def seed_courses(db = Depends(get_db)):
    """
    One-time endpoint: seed courses table from cleaned_courses.csv.
    Call once after DB setup: POST /api/courses/seed
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
        
    with db.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM courses")
        count = cursor.fetchone()['count']
        if count > 0:
            return {"message": "Courses already seeded", "count": count}

    courses_df = get_courses_df()
    if courses_df is None:
        raise HTTPException(status_code=500, detail="ML artefacts not loaded")

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
    with db.cursor() as cursor:
        for _, row in courses_df.iterrows():
            cols = []
            vals = []
            for csv_col, db_col in col_map.items():
                if csv_col in row:
                    val = row[csv_col]
                    cols.append(db_col)
                    vals.append(None if pd.isna(val) else val)
                    
            placeholders = ', '.join(['%s'] * len(vals))
            col_names = ', '.join(cols)
            cursor.execute(f"INSERT INTO courses ({col_names}) VALUES ({placeholders})", tuple(vals))
            inserted += 1

    db.commit()
    return {"message": f"Seeded {inserted} courses"}
