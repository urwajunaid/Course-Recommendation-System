"""
config.py — reads from .env file (located in backend/)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MySQL
    DB_HOST     : str = "localhost"
    DB_PORT     : int = 3306
    DB_USER     : str = "root"
    DB_PASSWORD : str = "Rafey@1308"
    DB_NAME     : str = "course_recommender"

    # JWT
    SECRET_KEY  : str = "change-me-in-production-use-a-long-random-string"
    ALGORITHM   : str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # ML artefacts — relative path is resolved from backend/ directory
    # Points to ML_model/data/ which contains .pkl, .keras, .csv files
    DATA_DIR    : str = "../ML_model/data"

    class Config:
        env_file = ".env"


settings = Settings()
