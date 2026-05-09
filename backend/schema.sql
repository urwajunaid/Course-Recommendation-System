-- ============================================================
--  Course Recommendation System — MySQL Schema
-- ============================================================

CREATE DATABASE IF NOT EXISTS course_recommender;
USE course_recommender;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    ncf_user_id  VARCHAR(10) UNIQUE NOT NULL,   -- e.g. U0001
    username     VARCHAR(80)  UNIQUE NOT NULL,
    email        VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Courses table (seeded from cleaned_courses.csv)
CREATE TABLE IF NOT EXISTS courses (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    course_name     VARCHAR(300) NOT NULL,
    company_name    VARCHAR(150),
    difficulty      VARCHAR(50),
    ratings         FLOAT,
    reviews         VARCHAR(50),
    type_of_certificate VARCHAR(100),
    duration        VARCHAR(100),
    skills          TEXT
);

-- User–course interaction log (feeds back into NCF retraining)
CREATE TABLE IF NOT EXISTS interactions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT NOT NULL,
    course_id   INT NOT NULL,
    rating      FLOAT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)   REFERENCES users(id)   ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    UNIQUE KEY uq_user_course (user_id, course_id)
);

-- Recommendation log (what was shown to whom)
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT,
    query_course    VARCHAR(300) NOT NULL,
    recommended     JSON NOT NULL,        -- list of course names returned
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Index for fast lookups
CREATE INDEX idx_interactions_user   ON interactions(user_id);
CREATE INDEX idx_interactions_course ON interactions(course_id);
CREATE INDEX idx_courses_name        ON courses(course_name(100));
