"""
============================================================
  Evaluation — Ranking Metrics (Precision, Recall, NDCG, Coverage)
  Phase 4 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Run: python eval/ranking_metrics.py
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
    return model, test_df, user_enc, course_enc


def calculate_ranking_metrics(model, test_df, course_enc, user_enc, k=10, num_users=100):
    precisions        = []
    recalls           = []
    ndcgs             = []
    recommended_items = set()
    all_unique_courses = set(course_enc.classes_)

    eval_users = test_df['user_id'].unique()[:num_users]

    for user_id in eval_users:
        actual_liked = set(
            test_df[
                (test_df['user_id'] == user_id) & (test_df['rating'] >= 4.0)
            ]['course_id'].tolist()
        )
        if not actual_liked:
            continue
        if user_id not in user_enc.classes_:
            continue

        u_idx          = user_enc.transform([user_id])[0]
        all_course_ids = course_enc.classes_
        c_indices      = course_enc.transform(all_course_ids)

        user_input = np.full(len(c_indices), u_idx)
        preds      = model.predict([user_input, c_indices], verbose=0).flatten()

        top_indices   = np.argsort(preds)[::-1][:k]
        top_k_items   = all_course_ids[top_indices]
        recommended_items.update(top_k_items)

        hits = len(set(top_k_items) & actual_liked)
        precisions.append(hits / k)
        recalls.append(hits / len(actual_liked) if len(actual_liked) > 0 else 0)

        dcg  = 0.0
        idcg = 0.0
        for i in range(min(len(actual_liked), k)):
            idcg += 1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k_items):
            if item in actual_liked:
                dcg += 1.0 / np.log2(i + 2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    coverage = len(recommended_items) / len(all_unique_courses)

    print(f"--- Ranking Metrics @ {k} (Evaluated on {len(precisions)} users) ---")
    print(f"✅ Precision@{k}: {np.mean(precisions):.4f}")
    print(f"✅ Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"✅ NDCG@{k}:      {np.mean(ndcgs):.4f}")
    print(f"✅ Coverage@{k}:  {coverage:.4f} ({len(recommended_items)}/{len(all_unique_courses)} unique courses)")

    return {
        "precision": np.mean(precisions),
        "recall":    np.mean(recalls),
        "ndcg":      np.mean(ndcgs),
        "coverage":  coverage,
    }


if __name__ == "__main__":
    model, test_df, user_enc, course_enc = load_artifacts()
    calculate_ranking_metrics(model, test_df, course_enc, user_enc, k=10)