"""
============================================================
  Evaluation — Average Score: Seen vs Unseen Courses
  Phase 4 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Run: python eval/test_avg_score.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "ML_model", "data")

INTERACTIONS_CSV = os.path.join(DATA_DIR, "synthetic_interactions.csv")
MODEL_PATH       = os.path.join(DATA_DIR, "ncf_model.keras")
USER_ENC_PATH    = os.path.join(DATA_DIR, "ncf_user_encoder.pkl")
COURSE_ENC_PATH  = os.path.join(DATA_DIR, "ncf_course_encoder.pkl")

R_MIN, R_MAX = 1.0, 5.0


def load_artifacts():
    print("📂 Loading artifacts …")
    interactions_df = pd.read_csv(INTERACTIONS_CSV)
    if "Unnamed: 0" in interactions_df.columns:
        interactions_df.drop(columns=["Unnamed: 0"], inplace=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(USER_ENC_PATH,   "rb") as f:
        user_enc = pickle.load(f)
    with open(COURSE_ENC_PATH, "rb") as f:
        course_enc = pickle.load(f)

    print(f"  ✅ Model loaded      : {MODEL_PATH}")
    print(f"  ✅ Interactions      : {len(interactions_df)} rows\n")
    return model, interactions_df, user_enc, course_enc


def test_avg_score(model, interactions_df, user_enc, course_enc,
                   test_user_id_int=0, seen_sample=10, unseen_sample=20):
    actual_user_id_str = f'U{test_user_id_int:04d}'

    if actual_user_id_str not in user_enc.classes_:
        print(f"⚠️  User {actual_user_id_str} was not seen during training. Aborting.")
        return

    all_course_ids = set(course_enc.classes_)
    seen_courses   = (
        interactions_df[interactions_df['user_id'] == actual_user_id_str]['course_id']
        .tolist()
    )
    unseen_courses = [c for c in all_course_ids if c not in seen_courses][:unseen_sample]

    u_encoded = user_enc.transform([actual_user_id_str])[0]

    seen_scores = []
    for c in seen_courses[:seen_sample]:
        if c not in course_enc.classes_:
            continue
        co_encoded = course_enc.transform([c])[0]
        pred_norm  = model.predict(
            [np.array([u_encoded]), np.array([co_encoded])], verbose=0
        )[0][0]
        seen_scores.append(pred_norm * (R_MAX - R_MIN) + R_MIN)

    unseen_scores = []
    for c in unseen_courses:
        if c not in course_enc.classes_:
            continue
        co_encoded = course_enc.transform([c])[0]
        pred_norm  = model.predict(
            [np.array([u_encoded]), np.array([co_encoded])], verbose=0
        )[0][0]
        unseen_scores.append(pred_norm * (R_MAX - R_MIN) + R_MIN)

    if not seen_scores or not unseen_scores:
        print("⚠️  Not enough data to compare seen vs unseen scores.")
        return

    print(f"User: {actual_user_id_str}")
    print(f"  Seen courses sampled  : {len(seen_scores)}")
    print(f"  Unseen courses sampled: {len(unseen_scores)}")
    print(f"\nAvg score for SEEN courses   : {np.mean(seen_scores):.4f}")
    print(f"Avg score for UNSEEN courses : {np.mean(unseen_scores):.4f}")

    if np.mean(seen_scores) > np.mean(unseen_scores):
        print("✅ Model correctly scores seen courses higher")
    else:
        print("❌ Model is not learning user preferences")


if __name__ == "__main__":
    model, interactions_df, user_enc, course_enc = load_artifacts()
    test_avg_score(model, interactions_df, user_enc, course_enc, test_user_id_int=0)