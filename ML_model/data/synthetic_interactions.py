"""
============================================================
  Synthetic User-Course Interaction Data Generator v2
  Phase 2 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
FIXES applied vs v1:
  ✅ User personas with specific skill preferences
  ✅ Full 1–5 rating range (not just 4–5)
  ✅ Strong negatives for non-matching courses
  ✅ Matching logic: skill overlap → high rating
  ✅ True contrast: liked vs disliked items
  ✅ More interactions per user for denser matrix

Inputs  : cleaned_courses.csv
Outputs :
  synthetic_interactions.csv
  ncf_interactions.csv
  interaction_stats.txt

Run: python synthetic_interactions.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ── 0. Config ─────────────────────────────────────────────
DATA_PATH   = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv"
OUTPUT_DIR  = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data"

NUM_USERS        = 500
MIN_INTERACTIONS = 15
MAX_INTERACTIONS = 30
RANDOM_SEED      = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. User Persona Definitions ───────────────────────────
PERSONAS = {
    "AI_Learner": {
        "preferred_skills": [
            "machine learning", "python programming", "data analysis",
            "algorithms", "machine learning algorithms",
            "applied machine learning", "data visualization",
            "probability & statistics", "statistical analysis",
            "exploratory data analysis", "statistical programming",
            "regression", "databases", "sql"
        ],
        "avoided_skills": [
            "leadership and management", "marketing", "finance",
            "entrepreneurship", "organizational development", "strategy"
        ],
        "preferred_diff": ["intermediate", "advanced"],
        "weight": 0.25
    },
    "Business_Learner": {
        "preferred_skills": [
            "leadership and management", "strategy", "business analysis",
            "finance", "marketing", "strategy and operations",
            "decision making", "planning", "entrepreneurship",
            "organizational development", "financial analysis",
            "risk management", "project management", "communication"
        ],
        "avoided_skills": [
            "machine learning", "python programming", "algorithms",
            "databases", "sql", "computer programming", "cloud computing"
        ],
        "preferred_diff": ["beginner", "intermediate"],
        "weight": 0.25
    },
    "Software_Engineer": {
        "preferred_skills": [
            "computer programming", "algorithms", "data structures",
            "software engineering", "programming principles",
            "cloud computing", "databases", "sql",
            "python programming", "problem solving", "mathematics"
        ],
        "avoided_skills": [
            "leadership and management", "marketing", "finance",
            "entrepreneurship", "organizational development"
        ],
        "preferred_diff": ["intermediate", "advanced"],
        "weight": 0.20
    },
    "Data_Analyst": {
        "preferred_skills": [
            "data analysis", "data visualization", "sql",
            "probability & statistics", "general statistics",
            "data management", "exploratory data analysis",
            "business analysis", "regression", "statistical analysis",
            "python programming", "decision making"
        ],
        "avoided_skills": [
            "cloud computing", "software engineering",
            "entrepreneurship", "marketing"
        ],
        "preferred_diff": ["beginner", "intermediate"],
        "weight": 0.20
    },
    "General_Learner": {
        "preferred_skills": [
            "communication", "critical thinking", "problem solving",
            "collaboration", "human learning", "planning",
            "decision making", "strategy"
        ],
        "avoided_skills": [
            "machine learning", "algorithms", "databases",
            "software engineering", "data structures"
        ],
        "preferred_diff": ["beginner", "mixed"],
        "weight": 0.10
    },
}

PERSONA_NAMES   = list(PERSONAS.keys())
PERSONA_WEIGHTS = [PERSONAS[p]["weight"] for p in PERSONA_NAMES]


# ── 2. Load dataset ───────────────────────────────────────
print("\n📂 Loading cleaned dataset …")
df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df = df.reset_index(drop=True)
df["course_id"]    = df.index
df["skills_clean"] = df["Skills"].fillna("").str.lower()
print(f"  Courses loaded : {len(df)}")


# ── 3. Skill match scoring ────────────────────────────────
def skill_match_score(course_skills, preferred, avoided):
    course_list = [s.strip() for s in course_skills.split(",")]
    pref_hits   = sum(1 for s in preferred
                      if any(s in cs for cs in course_list))
    avoid_hits  = sum(1 for s in avoided
                      if any(s in cs for cs in course_list))
    pref_score  = pref_hits  / max(len(preferred), 1)
    avoid_score = avoid_hits / max(len(avoided),   1)
    return float(np.clip(pref_score - 0.7 * avoid_score, 0.0, 1.0))


# ── 4. Pre-compute match scores ───────────────────────────
print("⚙️  Pre-computing skill match scores …")
match_scores = {}
for persona_name, persona in PERSONAS.items():
    match_scores[persona_name] = df["skills_clean"].apply(
        lambda s: skill_match_score(
            s, persona["preferred_skills"], persona["avoided_skills"]
        )
    ).values
print("  ✅ Done\n")


# ── 5. Generate interactions ──────────────────────────────
print(f"⚙️  Generating interactions for {NUM_USERS} users …")
interactions = []
start_date   = datetime(2024, 1, 1)

for user_id in range(NUM_USERS):
    persona_name = np.random.choice(PERSONA_NAMES, p=PERSONA_WEIGHTS)
    persona      = PERSONAS[persona_name]
    scores       = match_scores[persona_name]

    # difficulty preference boost
    diff_boost = np.ones(len(df))
    for i, diff in enumerate(df["Difficulty"].fillna("").str.lower()):
        if diff in [d.lower() for d in persona["preferred_diff"]]:
            diff_boost[i] = 1.5

    sample_weights = (scores + 0.1) * diff_boost
    sample_weights = sample_weights / sample_weights.sum()

    n_interactions = np.random.randint(MIN_INTERACTIONS, MAX_INTERACTIONS + 1)
    course_ids     = np.random.choice(
        df["course_id"].values,
        size=min(n_interactions, len(df)),
        replace=False,
        p=sample_weights
    )

    for course_id in course_ids:
        course_row = df[df["course_id"] == course_id].iloc[0]
        match      = scores[course_id]

        # rating based on match level
        if match >= 0.5:
            base_rating = np.random.uniform(3.8, 5.0)
        elif match >= 0.2:
            base_rating = np.random.uniform(2.0, 3.8)
        else:
            base_rating = np.random.uniform(1.0, 2.5)

        noise       = np.random.normal(0, 0.3)
        user_rating = float(np.clip(round(base_rating + noise, 1), 1.0, 5.0))

        if user_rating >= 4.5:
            itype = "completed"
        elif user_rating >= 3.5:
            itype = np.random.choice(["completed", "in_progress"], p=[0.6, 0.4])
        elif user_rating >= 2.5:
            itype = np.random.choice(["in_progress", "dropped"], p=[0.3, 0.7])
        else:
            itype = "dropped"

        days_offset = np.random.randint(0, 365)
        timestamp   = start_date + timedelta(days=int(days_offset))

        interactions.append({
            "user_id"           : f"U{user_id:04d}",
            "course_id"         : int(course_id),
            "course_name"       : course_row["Course_Name"],
            "difficulty"        : course_row["Difficulty"],
            "rating"            : user_rating,
            "skill_match_score" : round(float(match), 3),
            "interaction_type"  : itype,
            "timestamp"         : timestamp.strftime("%Y-%m-%d"),
            "user_persona"      : persona_name,
        })

interactions_df = pd.DataFrame(interactions)
print(f"  Total interactions : {len(interactions_df)}\n")


# ── 6. Implicit feedback ──────────────────────────────────
interactions_df["implicit_feedback"] = (
    interactions_df["rating"] >= 3.0).astype(int)

# flip 10% of positives to negative (simulates drop-off)
pos_idx  = interactions_df[interactions_df["implicit_feedback"] == 1].index
flip_idx = np.random.choice(pos_idx,
                             size=int(len(pos_idx) * 0.10),
                             replace=False)
interactions_df.loc[flip_idx, "implicit_feedback"] = 0


# ── 7. Validation ─────────────────────────────────────────
print("✅ Validation …")
print(f"  Unique users       : {interactions_df['user_id'].nunique()}")
print(f"  Unique courses     : {interactions_df['course_id'].nunique()}")
print(f"  Total interactions : {len(interactions_df)}")
print(f"  Avg per user       : "
      f"{len(interactions_df)/interactions_df['user_id'].nunique():.1f}")
print(f"\n  Rating distribution:")
print(interactions_df["rating"].describe().round(2).to_string())
print(f"\n  Rating std dev     : {interactions_df['rating'].std():.3f}  "
      f"(target > 1.0)")
print(f"\n  Implicit feedback:")
vc = interactions_df["implicit_feedback"].value_counts()
print(f"  Positive (1) : {vc.get(1,0)}  "
      f"({vc.get(1,0)/len(interactions_df)*100:.1f}%)")
print(f"  Negative (0) : {vc.get(0,0)}  "
      f"({vc.get(0,0)/len(interactions_df)*100:.1f}%)")
print(f"\n  Avg rating by persona:")
print(interactions_df.groupby("user_persona")["rating"]
      .mean().round(2).to_string())


# ── 8. Save ───────────────────────────────────────────────
print("\n💾 Saving …")

out1 = os.path.join(OUTPUT_DIR, "synthetic_interactions.csv")
interactions_df.to_csv(out1, index=False)
print(f"  ✅ Full log     → {out1}")

ncf_df = interactions_df[["user_id", "course_id",
                            "rating", "implicit_feedback"]]
out2   = os.path.join(OUTPUT_DIR, "ncf_interactions.csv")
ncf_df.to_csv(out2, index=False)
print(f"  ✅ NCF CSV      → {out2}")

stats_path = os.path.join(OUTPUT_DIR, "interaction_stats.txt")
with open(stats_path, "w") as f:
    f.write("SYNTHETIC INTERACTION STATS v2\n")
    f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"Users      : {interactions_df['user_id'].nunique()}\n")
    f.write(f"Courses    : {interactions_df['course_id'].nunique()}\n")
    f.write(f"Total      : {len(interactions_df)}\n\n")
    f.write("Rating Stats:\n")
    f.write(interactions_df["rating"].describe().round(2).to_string())
    f.write("\n\nPersona Distribution:\n")
    f.write(interactions_df["user_persona"].value_counts().to_string())
    f.write("\n\nImplicit Feedback:\n")
    f.write(interactions_df["implicit_feedback"].value_counts().to_string())
print(f"  ✅ Stats        → {stats_path}")


# ── 9. Summary ────────────────────────────────────────────
print("\n" + "="*55)
print("  📋 SUMMARY")
print("="*55)
print(f"  Users              : {NUM_USERS}")
print(f"  Personas           : {len(PERSONAS)}")
print(f"  Total interactions : {len(interactions_df)}")
print(f"  Rating range       : "
      f"{interactions_df['rating'].min()} – {interactions_df['rating'].max()}")
print(f"  Rating std dev     : {interactions_df['rating'].std():.3f}")
print(f"  Positive feedback  : "
      f"{(interactions_df['implicit_feedback']==1).mean()*100:.1f}%")
print(f"  Negative feedback  : "
      f"{(interactions_df['implicit_feedback']==0).mean()*100:.1f}%")
print("="*55)
print("\n✅ Done! Now re-run ncf_model.py with this improved data.\n")
