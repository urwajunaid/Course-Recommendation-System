"""
routers/auth.py — Register, Login, Me
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from database import get_db, User
from security import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────

class RegisterRequest(BaseModel):
    username : str
    email    : EmailStr
    password : str


class LoginRequest(BaseModel):
    email    : EmailStr
    password : str


class TokenResponse(BaseModel):
    access_token : str
    token_type   : str = "bearer"
    user_id      : int
    username     : str
    ncf_user_id  : str


# ── Helpers ───────────────────────────────────────────────

def _generate_ncf_id(db) -> str:
    """Generate next available NCF user ID like U0042."""
    with db.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM users")
        count = cursor.fetchone()['count']
    return f"U{(count + 1):04d}"


# ── Endpoints ─────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db = Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
        
    with db.cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE email = %s", (body.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
            
        cursor.execute("SELECT id FROM users WHERE username = %s", (body.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already taken")

        ncf_user_id = _generate_ncf_id(db)
        hashed_pw = hash_password(body.password)
        
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, ncf_user_id)
            VALUES (%s, %s, %s, %s)
        """, (body.username, body.email, hashed_pw, ncf_user_id))
        user_id = cursor.lastrowid
        
    db.commit()

    token = create_access_token({"sub": str(user_id)})
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        username=body.username,
        ncf_user_id=ncf_user_id,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db = Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
        
    with db.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE email = %s", (body.email,))
        row = cursor.fetchone()
        
    if not row or not verify_password(body.password, row['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = User(row)
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        ncf_user_id=user.ncf_user_id,
    )


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id"         : current_user.id,
        "username"   : current_user.username,
        "email"      : current_user.email,
        "ncf_user_id": current_user.ncf_user_id,
        "created_at" : current_user.created_at,
    }
