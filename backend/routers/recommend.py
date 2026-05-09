"""
routers/recommend.py
  GET  /api/recommend?course=...&top_n=5          — guest or authed
  GET  /api/recommend/personalized?top_n=5        — requires auth
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from database import get_db, get_optional_db
from security import get_current_user, get_optional_user
from ml.loader import get_recommender
from database import User

router = APIRouter()


@router.get("/")
def recommend(
    course      : str            = Query(..., description="Seed course name"),
    top_n       : int            = Query(5, ge=1, le=20),
    current_user: Optional[User] = Depends(get_optional_user),
    db                          = Depends(get_optional_db),
):
    """
    Hybrid recommendation.
    - If authenticated: uses user's NCF ID for personalised re-ranking.
    - If guest: content-based only.
    """
    recommender = get_recommender()
    user_ncf_id = current_user.ncf_user_id if current_user else None

    try:
        results = recommender.recommend(
            course_name = course,
            user_id     = user_ncf_id,
            top_n       = top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Log the recommendation (best-effort — skip if DB unavailable)
    if db:
        try:
            import json
            with db.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO recommendation_logs (user_id, query_course, recommended)
                    VALUES (%s, %s, %s)
                """, (current_user.id if current_user else None, course, json.dumps([r["course_name"] for r in results])))
            db.commit()
        except Exception:
            pass  # DB unavailable — skip logging, don't fail the request

    return {
        "query_course" : course,
        "user_id"      : current_user.ncf_user_id if current_user else None,
        "mode"         : "hybrid" if current_user else "content-based",
        "results"      : results,
    }


@router.get("/personalized")
def personalized(
    top_n        : int  = Query(10, ge=1, le=20),
    current_user : User = Depends(get_current_user),
    db           = Depends(get_db),
):
    """
    Personalised home feed: picks seed courses from the user's top-rated
    interactions, falls back to popular courses if none exist.
    """
    recommender = get_recommender()

    interactions = []
    if db:
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT c.course_name
                FROM interactions i
                JOIN courses c ON i.course_id = c.id
                WHERE i.user_id = %s
                ORDER BY i.rating DESC
                LIMIT 3
            """, (current_user.id,))
            interactions = cursor.fetchall()

    if interactions:
        seeds = [i['course_name'] for i in interactions]
    else:
        # Cold-start: pick random popular seeds
        seeds = ["machine learning", "python for everybody", "google data analytics"]

    all_results = []
    seen = set()
    for seed in seeds[:2]:
        try:
            recs = recommender.recommend(
                course_name = seed,
                user_id     = current_user.ncf_user_id,
                top_n       = top_n,
            )
            for r in recs:
                if r["course_name"] not in seen:
                    seen.add(r["course_name"])
                    all_results.append(r)
        except ValueError:
            continue

    return {
        "user"    : current_user.username,
        "seeds"   : seeds[:2],
        "results" : all_results[:top_n],
    }
