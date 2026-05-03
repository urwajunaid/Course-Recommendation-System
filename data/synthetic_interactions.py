"""
============================================================
  Synthetic User-Course Interaction Data Generator
  Phase 2 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Inputs  : cleaned_courses.csv
Outputs :
  synthetic_interactions.csv   — user-course interaction log
  interaction_stats.txt        — summary statistics

Role    : NCF (Neural Collaborative Filtering) requires
          user-course interaction history to learn
          preference patterns. Since no real user data
          exists, we simulate realistic interactions based
          on course ratings, difficulty, and skill overlap.

Run: python synthetic_interactions.py
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

# ── 0. Config ─────────────────────────────────────────────
DATA_PATH   = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv"
OUTPUT_DIR  = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data"

# Interaction generation parameters
NUM_USERS           = 500     # synthetic users to generate
MIN_INTERACTIONS    = 5       # min courses each user interacts with
MAX_INTERACTIONS    = 20      # max courses each user interacts with
RANDOM_SEED         = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load cleaned dataset ───────────────────────────────
print("\n📂 Loading cleaned dataset …")
df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df = df.reset_index(drop=True)
df["course_id"] = df.index         # numeric ID for each course

print(f"  Courses loaded : {len(df)}")
print(f"  Columns        : {df.columns.tolist()}\n")


# ── 2. Build difficulty preference profiles ───────────────
#       Each user is assigned a preferred difficulty level
#       so interactions feel realistic (beginners mostly
#       take beginner courses, etc.)
DIFFICULTY_LEVELS = df["Difficulty"].dropna().unique().tolist()

# weight distribution — beginners dominate (mirrors real data)
DIFF_WEIGHTS = []
for d in DIFFICULTY_LEVELS:
    count = (df["Difficulty"] == d).sum()
    DIFF_WEIGHTS.append(count)
DIFF_WEIGHTS = np.array(DIFF_WEIGHTS) / sum(DIFF_WEIGHTS)

print("📋 Difficulty distribution in dataset:")
for d, w in zip(DIFFICULTY_LEVELS, DIFF_WEIGHTS):
    print(f"  {d:<20} : {w*100:.1f}%")
print()


# ── 3. Rating-based interaction probability ───────────────
#       Higher-rated courses are more likely to be
#       interacted with — simulates popularity bias
rating_col = df["Ratings"].fillna(df["Ratings"].median())
# normalize to probability weights
rating_weights = (rating_col - rating_col.min() + 0.1)
rating_weights = rating_weights / rating_weights.sum()


# ── 4. Generate interactions ──────────────────────────────
print(f"⚙️  Generating interactions for {NUM_USERS} users …")

interactions = []
start_date = datetime(2024, 1, 1)

for user_id in range(NUM_USERS):

    # ignore dataset imbalance — assign difficulties evenly across users
    balanced_weights = np.ones(len(DIFFICULTY_LEVELS)) / len(DIFFICULTY_LEVELS)
    user_difficulty = np.random.choice(DIFFICULTY_LEVELS, p=balanced_weights)

    # how many courses this user interacts with
    n_interactions = np.random.randint(MIN_INTERACTIONS, MAX_INTERACTIONS + 1)

    # build course weights for this user:
    # boost courses matching their preferred difficulty
    user_weights = rating_weights.copy()
    difficulty_mask = (df["Difficulty"] == user_difficulty).values
    user_weights[difficulty_mask] *= 1.5     
    user_weights = user_weights / user_weights.sum()

    # sample courses without replacement
    n_sample = min(n_interactions, len(df))
    course_ids = np.random.choice(
        df["course_id"].values,
        size=n_sample,
        replace=False,
        p=user_weights
    )

    for course_id in course_ids:
        course_row   = df[df["course_id"] == course_id].iloc[0]
        base_rating  = course_row["Ratings"] if not pd.isna(course_row["Ratings"]) \
                       else 4.0

        # implicit rating: base course rating ± small user noise
        noise = np.random.normal(0, 0.6)   # wider spread
        # also randomly assign some users as "harsh raters"
        if np.random.random() < 0.2:       # 20% of users rate harshly
            noise -= 1.0
        user_rating  = float(np.clip(round(base_rating + noise, 1), 1.0, 5.0))

        # simulate interaction type based on rating
        if user_rating >= 4.5:
            interaction_type = "completed"
        elif user_rating >= 3.5:
            interaction_type = np.random.choice(
                ["completed", "in_progress"], p=[0.6, 0.4])
        elif user_rating >= 2.5:
            interaction_type = np.random.choice(
                ["in_progress", "dropped"], p=[0.4, 0.6])
        else:
            interaction_type = "dropped"

        # simulate a random interaction timestamp
        days_offset  = np.random.randint(0, 365)
        timestamp    = start_date + timedelta(days=int(days_offset))

        interactions.append({
            "user_id"          : f"U{user_id:04d}",
            "course_id"        : int(course_id),
            "course_name"      : course_row["Course_Name"],
            "difficulty"       : course_row["Difficulty"],
            "rating"           : user_rating,
            "interaction_type" : interaction_type,
            "timestamp"        : timestamp.strftime("%Y-%m-%d"),
            "user_difficulty_pref" : user_difficulty,
        })

interactions_df = pd.DataFrame(interactions)
print(f"  Total interactions generated : {len(interactions_df)}\n")


# ── 5. Add implicit feedback column ──────────────────────
#       Binary: 1 = positive interaction, 0 = negative
#       Used directly by NCF as the training label
# lower threshold + inject random negative samples (20% of positives)
interactions_df["implicit_feedback"] = interactions_df["rating"].apply(
    lambda r: 1 if r >= 4.0 else 0   # stricter threshold
)

# randomly flip 15% of positives to negatives (simulates abandoned courses)
pos_idx = interactions_df[interactions_df["implicit_feedback"] == 1].index
flip_idx = np.random.choice(pos_idx, size=int(len(pos_idx) * 0.15), replace=False)
interactions_df.loc[flip_idx, "implicit_feedback"] = 0


# ── 6. Validation checks ──────────────────────────────────
print("✅ Validation checks …")
print(f"  Unique users          : {interactions_df['user_id'].nunique()}")
print(f"  Unique courses        : {interactions_df['course_id'].nunique()}")
print(f"  Total interactions    : {len(interactions_df)}")
print(f"  Avg interactions/user : "
      f"{len(interactions_df)/interactions_df['user_id'].nunique():.1f}")
print(f"  Positive feedback (1) : "
      f"{(interactions_df['implicit_feedback']==1).sum()} "
      f"({(interactions_df['implicit_feedback']==1).mean()*100:.1f}%)")
print(f"  Negative feedback (0) : "
      f"{(interactions_df['implicit_feedback']==0).sum()} "
      f"({(interactions_df['implicit_feedback']==0).mean()*100:.1f}%)")
print(f"\n  Interaction types:")
print(interactions_df["interaction_type"].value_counts().to_string())
print(f"\n  Rating distribution:")
print(interactions_df["rating"].describe().round(2).to_string())


# ── 7. Save outputs ───────────────────────────────────────
print("\n💾 Saving outputs …")

# 7a. Full interaction log
interactions_path = os.path.join(OUTPUT_DIR, "synthetic_interactions.csv")
interactions_df.to_csv(interactions_path, index=False)
print(f"  ✅ Interactions CSV → {interactions_path}")

# 7b. NCF-ready minimal version (user_id, course_id, implicit_feedback only)
ncf_df = interactions_df[["user_id", "course_id", "rating", "implicit_feedback"]]
ncf_path = os.path.join(OUTPUT_DIR, "ncf_interactions.csv")
ncf_df.to_csv(ncf_path, index=False)
print(f"  ✅ NCF-ready CSV    → {ncf_path}")

# 7c. Stats summary text file
stats_path = os.path.join(OUTPUT_DIR, "interaction_stats.txt")
with open(stats_path, "w") as f:
    f.write("="*50 + "\n")
    f.write("  SYNTHETIC INTERACTION DATA STATISTICS\n")
    f.write("  Course Recommendation System | Phase 2\n")
    f.write("="*50 + "\n\n")
    f.write(f"Generated on     : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Random seed      : {RANDOM_SEED}\n\n")
    f.write(f"Total users      : {interactions_df['user_id'].nunique()}\n")
    f.write(f"Total courses    : {interactions_df['course_id'].nunique()}\n")
    f.write(f"Total interactions: {len(interactions_df)}\n")
    f.write(f"Avg per user     : {len(interactions_df)/interactions_df['user_id'].nunique():.1f}\n")
    f.write(f"Sparsity         : "
            f"{100*(1 - len(interactions_df)/(interactions_df['user_id'].nunique()*len(df))):.1f}%\n\n")
    f.write("Interaction Types:\n")
    f.write(interactions_df["interaction_type"].value_counts().to_string())
    f.write("\n\nRating Distribution:\n")
    f.write(interactions_df["rating"].describe().round(2).to_string())
    f.write("\n\nImplicit Feedback:\n")
    f.write(interactions_df["implicit_feedback"].value_counts().to_string())
print(f"  ✅ Stats summary    → {stats_path}")


# ── 8. Summary ────────────────────────────────────────────
print("\n" + "="*55)
print("  📋 INTERACTION GENERATION SUMMARY")
print("="*55)
print(f"  Users              : {NUM_USERS}")
print(f"  Courses in pool    : {len(df)}")
print(f"  Total interactions : {len(interactions_df)}")
print(f"  NCF training rows  : {len(ncf_df)}")
print(f"  Files saved to     : {OUTPUT_DIR}")
print("="*55)
print("\n✅ Synthetic interaction generation complete!\n")
print("📌 Next step: Use ncf_interactions.csv to train the NCF model in Phase 4.")
