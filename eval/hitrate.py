"""
============================================================
  Evaluation — Hit Rate @ K
  Phase 4 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Run: python eval/hitrate.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    _, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(USER_ENC_PATH,   "rb") as f:
        user_enc = pickle.load(f)
    with open(COURSE_ENC_PATH, "rb") as f:
        course_enc = pickle.load(f)

    print(f"  ✅ Model loaded      : {MODEL_PATH}")
    print(f"  ✅ Interactions      : {len(interactions_df)} rows")
    print(f"  ✅ Test split        : {len(test_df)} rows\n")
    return model, interactions_df, test_df, user_enc, course_enc


def hit_rate_at_k(model, test_df, interactions_df, user_enc, course_enc,
                  k=10, num_users=50):
    hits               = 0
    num_users_evaluated = 0

    for user_id_str in test_df['user_id'].unique()[:num_users]:
        actual = test_df[
            (test_df['user_id'] == user_id_str) & (test_df['rating'] >= 4.0)
        ]['course_id'].tolist()

        if not actual:
            continue
        if user_id_str not in user_enc.classes_:
            continue

        encoded_user_idx = user_enc.transform([user_id_str])[0]

        all_original_course_ids    = course_enc.classes_
        all_encoded_course_indices = course_enc.transform(all_original_course_ids)

        user_input_array   = np.full(len(all_encoded_course_indices), encoded_user_idx)
        course_input_array = np.array(all_encoded_course_indices)

        predictions_norm = model.predict(
            [user_input_array, course_input_array], verbose=0
        ).flatten()
        predictions = predictions_norm * (R_MAX - R_MIN) + R_MIN

        all_scores = list(zip(all_original_course_ids, predictions))
        top_k      = sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]
        top_k_ids  = [x[0] for x in top_k]

        if any(course_id in top_k_ids for course_id in actual):
            hits += 1

        num_users_evaluated += 1

    hit_rate = hits / num_users_evaluated if num_users_evaluated > 0 else 0.0
    print(f"Hit Rate @ {k}: {hit_rate:.4f}  ({hits}/{num_users_evaluated} users evaluated)")
    return hit_rate


if __name__ == "__main__":
    model, interactions_df, test_df, user_enc, course_enc = load_artifacts()
    hit_rate_at_k(model, test_df, interactions_df, user_enc, course_enc, k=10)