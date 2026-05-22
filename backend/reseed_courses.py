"""
reseed_courses.py
Force re-seeds the `courses` table from cleaned_courses.csv.

- Truncates the existing `courses` table (also clears interactions via CASCADE)
- Re-inserts all rows from the CSV
- Run from the backend/ directory:  python reseed_courses.py

NOTE: Truncating `courses` will also delete all `interactions` (ratings)
      because of the FOREIGN KEY … ON DELETE CASCADE constraint.
      Users themselves are NOT deleted.
"""

import sys
import os
sys.path.insert(0, '.')

from config import settings
import pymysql
import pandas as pd

# ── Locate CSV ─────────────────────────────────────────────────────────────────
data_dir = settings.DATA_DIR
if not os.path.isabs(data_dir):
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base, data_dir))

csv_path = os.path.join(data_dir, "cleaned_courses.csv")
print(f"[INFO] Reading CSV from: {csv_path}")

if not os.path.exists(csv_path):
    print(f"[ERROR] CSV not found at: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
df = df.reset_index(drop=True)
print(f"[INFO] CSV rows to seed: {len(df)}")

# ── Connect ─────────────────────────────────────────────────────────────────────
conn = pymysql.connect(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD,
    database=settings.DB_NAME,
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=False,
)
print(f"[INFO] Connected to database: {settings.DB_NAME}")

# ── Check before truncate ───────────────────────────────────────────────────────
with conn.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) as count FROM courses")
    before = cursor.fetchone()["count"]
    cursor.execute("SELECT COUNT(*) as count FROM interactions")
    interactions_before = cursor.fetchone()["count"]

print(f"[INFO] courses table currently has {before} rows")
print(f"[INFO] interactions table currently has {interactions_before} rows (will also be cleared)")

if before == len(df):
    resp = input(f"\n[WARN] Database already has {before} rows (matches CSV). Re-seed anyway? [y/N]: ").strip().lower()
    if resp != "y":
        print("[INFO] Aborted — no changes made.")
        conn.close()
        sys.exit(0)

# ── Truncate ────────────────────────────────────────────────────────────────────
print("\n[INFO] Truncating interactions and courses tables...")
with conn.cursor() as cursor:
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("TRUNCATE TABLE interactions")
    cursor.execute("TRUNCATE TABLE courses")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
conn.commit()
print("[OK] Tables truncated.")

# ── Column map (CSV column → DB column) ────────────────────────────────────────
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

# ── Insert in batches ───────────────────────────────────────────────────────────
BATCH_SIZE = 100
inserted = 0
batch_rows = []

with conn.cursor() as cursor:
    for _, row in df.iterrows():
        cols, vals = [], []
        for csv_col, db_col in col_map.items():
            if csv_col in df.columns:
                cols.append(db_col)
                val = row[csv_col]
                vals.append(None if pd.isna(val) else str(val) if db_col in ("reviews", "duration", "skills") else val)

        placeholders = ", ".join(["%s"] * len(vals))
        col_names    = ", ".join(cols)
        cursor.execute(
            f"INSERT INTO courses ({col_names}) VALUES ({placeholders})",
            tuple(vals),
        )
        inserted += 1

        if inserted % BATCH_SIZE == 0:
            conn.commit()
            print(f"  [OK] {inserted} / {len(df)} rows inserted...")

conn.commit()

# ── Verify ──────────────────────────────────────────────────────────────────────
with conn.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) as count FROM courses")
    after = cursor.fetchone()["count"]

conn.close()
print(f"\n[OK] Done! courses table now has {after} rows (seeded {inserted}).")
