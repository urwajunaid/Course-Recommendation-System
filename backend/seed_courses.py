"""
One-off script: seeds the courses table from cleaned_courses.csv.
Run from the backend/ directory:  python seed_courses.py
"""
import sys
sys.path.insert(0, '.')

from config import settings
import pymysql
import pandas as pd
import os

# ── Locate CSV ────────────────────────────────────────────
data_dir = settings.DATA_DIR
if not os.path.isabs(data_dir):
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base, data_dir))

csv_path = os.path.join(data_dir, "cleaned_courses.csv")
print(f"[INFO] Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
df = df.reset_index(drop=True)
print(f"[INFO] CSV rows: {len(df)}")
print(f"[INFO] Columns: {list(df.columns)}")

# ── Connect ───────────────────────────────────────────────
conn = pymysql.connect(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD,
    database=settings.DB_NAME,
    cursorclass=pymysql.cursors.DictCursor,
)

with conn.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) as count FROM courses")
    count = cursor.fetchone()["count"]
    print(f"[INFO] Existing courses in DB: {count}")

if count > 0:
    print("[INFO] Table already has data — skipping. Delete rows first if you want to re-seed.")
    conn.close()
    sys.exit(0)

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
BATCH_SIZE = 200
rows_batch = []

with conn.cursor() as cursor:
    for _, row in df.iterrows():
        cols, vals = [], []
        for csv_col, db_col in col_map.items():
            if csv_col in row:
                cols.append(db_col)
                val = row[csv_col]
                vals.append(None if pd.isna(val) else val)

        placeholders = ", ".join(["%s"] * len(vals))
        col_names = ", ".join(cols)
        cursor.execute(
            f"INSERT INTO courses ({col_names}) VALUES ({placeholders})",
            tuple(vals),
        )
        inserted += 1
        if inserted % BATCH_SIZE == 0:
            conn.commit()
            print(f"  [OK] Inserted {inserted} / {len(df)} rows...")

conn.commit()
conn.close()
print(f"\n[OK] Done! Seeded {inserted} courses into `courses` table.")
