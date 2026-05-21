"""
database.py — Raw SQL + MySQL (via PyMySQL)
"""

import pymysql
from datetime import datetime
from config import settings
import json

_db_available = False

class User:
    def __init__(self, row):
        self.id = row['id']
        self.ncf_user_id = row['ncf_user_id']
        self.username = row['username']
        self.email = row['email']
        self.password_hash = row['password_hash']
        self.created_at = row['created_at']

def get_connection():
    try:
        return pymysql.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception:
        return None

def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        if conn:
            conn.close()

def get_optional_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        if conn:
            conn.close()

def create_tables():
    # Connect without database to create it
    try:
        conn = pymysql.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.DB_NAME}")
        conn.close()
    except Exception as e:
        print(f"[WARN] Cannot connect to MySQL server: {e}")
        return
        
    conn = get_connection()
    if not conn:
        print("[WARN] DB unavailable, skipping table creation.")
        return

    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ncf_user_id VARCHAR(10) UNIQUE NOT NULL,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INT AUTO_INCREMENT PRIMARY KEY,
                course_name VARCHAR(300) NOT NULL,
                company_name VARCHAR(150),
                difficulty VARCHAR(50),
                ratings FLOAT,
                reviews VARCHAR(500),
                type_of_certificate VARCHAR(100),
                duration VARCHAR(200),
                skills TEXT,
                INDEX (course_name)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                course_id INT NOT NULL,
                rating FLOAT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
                UNIQUE (user_id, course_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                query_course VARCHAR(300) NOT NULL,
                recommended JSON NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
    conn.commit()
    conn.close()
