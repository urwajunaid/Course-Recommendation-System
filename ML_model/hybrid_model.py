"""
============================================================
  Hybrid Recommendation Model
  Phase 4 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Strategy : Content-Based Filtering retrieves top candidates
           via cosine similarity, then NCF re-ranks them
           using predicted ratings.

           Query Course
               │
               ▼
      Content-Based Filtering
      (cosine_sim.npy → top 50 similar courses)
               │
               ▼
      NCF Re-ranking
      (ncf_model.keras → predicted rating per candidate)
               │
               ▼
      Final Top-5 Recommendations

Inputs  :
  cleaned_courses.csv
  cosine_sim.npy
  course_indices.pkl
  ncf_model.keras
  ncf_user_encoder.pkl
  ncf_course_encoder.pkl

Outputs :
  hybrid_recommender.pkl     — saved hybrid recommender object
  hybrid_sample_output.csv   — sample recommendations for demo users

Run: python hybrid_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

# ── 0. Config ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data"
OUTPUT_DIR = DATA_DIR

TOP_N          = 5      # final recommendations to return
CBF_CANDIDATES = 50     # how many CBF results NCF re-ranks
R_MIN, R_MAX   = 1.0, 5.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*55)
print("  HYBRID RECOMMENDATION MODEL")
print("  CBF Retrieval → NCF Re-ranking")
print("="*55)


# ── 1. Load all artefacts ─────────────────────────────────
print("\n📂 Loading artefacts …")

# courses
courses = pd.read_csv(os.path.join(DATA_DIR, "cleaned_courses.csv"))
if "Unnamed: 0" in courses.columns:
    courses.drop(columns=["Unnamed: 0"], inplace=True)
courses = courses.reset_index(drop=True)
print(f"  ✅ Courses          : {len(courses)} rows")

# cosine similarity matrix
cosine_sim = np.load(os.path.join(DATA_DIR, "cosine_sim.npy"))
print(f"  ✅ Cosine sim       : {cosine_sim.shape}")

# course index map (course_name → row index)
with open(os.path.join(DATA_DIR, "course_indices.pkl"), "rb") as f:
    course_indices = pickle.load(f)
print(f"  ✅ Course indices   : {len(course_indices)} entries")

# NCF model
ncf_model = keras.models.load_model(os.path.join(DATA_DIR, "ncf_model.keras"))
print(f"  ✅ NCF model        : loaded")

# encoders
with open(os.path.join(DATA_DIR, "ncf_user_encoder.pkl"), "rb") as f:
    user_enc = pickle.load(f)
with open(os.path.join(DATA_DIR, "ncf_course_encoder.pkl"), "rb") as f:
    course_enc = pickle.load(f)
print(f"  ✅ Encoders         : loaded\n")


# ── 2. Hybrid Recommender Class ───────────────────────────
class HybridRecommender:
    """
    Two-stage recommender:
      Stage 1 — Content-Based Filtering retrieves top candidates
      Stage 2 — NCF re-ranks candidates by predicted user rating
    """

    def __init__(self, courses, cosine_sim, course_indices,
                 ncf_model, user_enc, course_enc,
                 top_n=5, cbf_candidates=50):
        self.courses         = courses
        self.cosine_sim      = cosine_sim
        self.course_indices  = course_indices
        self.ncf_model       = ncf_model
        self.user_enc        = user_enc
        self.course_enc      = course_enc
        self.top_n           = top_n
        self.cbf_candidates  = cbf_candidates

    # ── Stage 1: Content-Based retrieval ──────────────────
    def _cbf_candidates(self, course_name: str) -> pd.DataFrame:
        """Return top cbf_candidates similar courses via cosine similarity."""
        key = course_name.lower().strip()

        # fuzzy fallback: find closest match if exact key missing
        if key not in self.course_indices:
            matches = [c for c in self.course_indices.index
                       if key in c or c in key]
            if not matches:
                raise ValueError(
                    f"Course '{course_name}' not found. "
                    f"Check spelling or use a course name from the dataset."
                )
            key = matches[0]
            print(f"  ℹ️  Matched to: '{key}'")

        idx = self.course_indices[key]
        if hasattr(idx, '__len__'):   # multiple matches → take first
            idx = int(idx.iloc[0]) if hasattr(idx, 'iloc') else int(idx[0])
        else:
            idx = int(idx)
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: self.cbf_candidates + 1]  # exclude self

        candidate_indices = [i for i, _ in sim_scores]
        candidates        = self.courses.iloc[candidate_indices].copy()
        candidates["cbf_score"] = [s for _, s in sim_scores]
        candidates["original_idx"] = candidate_indices
        return candidates

    # ── Stage 2: NCF re-ranking ───────────────────────────
    def _ncf_rerank(self, candidates: pd.DataFrame,
                    user_id: str) -> pd.DataFrame:
        """Predict ratings for each candidate and sort descending."""
        # check if user exists in encoder
        if user_id not in self.user_enc.classes_:
            # cold-start: use average of all known users
            print(f"  ℹ️  User '{user_id}' not seen during training "
                  f"— using cold-start (average NCF score).")
            candidates["ncf_score"] = 3.5   # neutral fallback
            return candidates.sort_values(
                "cbf_score", ascending=False).head(self.top_n)

        user_idx     = self.user_enc.transform([user_id])[0]
        course_ids   = candidates["original_idx"].values

        # map dataset row indices → encoder indices
        valid_rows   = []
        valid_encs   = []
        for row_pos, cid in enumerate(course_ids):
            if cid in self.course_enc.classes_:
                enc_idx = self.course_enc.transform([cid])[0]
                valid_rows.append(row_pos)
                valid_encs.append(enc_idx)

        if not valid_encs:
            candidates["ncf_score"] = 3.5
            return candidates.sort_values(
                "cbf_score", ascending=False).head(self.top_n)

        user_arr  = np.full(len(valid_encs), user_idx)
        course_arr= np.array(valid_encs)

        preds_norm = self.ncf_model.predict(
            [user_arr, course_arr], verbose=0
        ).flatten()
        preds      = preds_norm * (R_MAX - R_MIN) + R_MIN

        # assign scores back
        candidates["ncf_score"] = np.nan
        for row_pos, score in zip(valid_rows, preds):
            candidates.iloc[row_pos,
                candidates.columns.get_loc("ncf_score")] = score

        candidates["ncf_score"] = candidates["ncf_score"].fillna(3.5)
        return candidates.sort_values(
            "ncf_score", ascending=False).head(self.top_n)

    # ── Public API ────────────────────────────────────────
    def recommend(self, course_name: str,
                  user_id: str = None,
                  verbose: bool = True) -> pd.DataFrame:
        """
        Get top-N hybrid recommendations.

        Parameters
        ----------
        course_name : str   — seed course (what the user is looking at)
        user_id     : str   — user ID (e.g. 'U0001'); None = content-only
        verbose     : bool  — print formatted results

        Returns
        -------
        pd.DataFrame with columns:
          Course_Name, Difficulty, Ratings, cbf_score, ncf_score
        """
        # Stage 1
        candidates = self._cbf_candidates(course_name)

        # Stage 2
        if user_id:
            results = self._ncf_rerank(candidates, user_id)
        else:
            # no user → rank by CBF score only
            results = candidates.sort_values(
                "cbf_score", ascending=False).head(self.top_n)
            results["ncf_score"] = np.nan

        output_cols = ["Course_Name", "Company_Name", "Difficulty",
                       "Ratings", "cbf_score", "ncf_score"]
        output_cols = [c for c in output_cols if c in results.columns]
        results     = results[output_cols].reset_index(drop=True)
        results.index += 1   # rank from 1

        if verbose:
            print(f"\n  📌 Query  : '{course_name}'")
            if user_id:
                print(f"  👤 User   : {user_id}")
            print(f"\n  {'Rank':<5} {'Course':<45} {'Diff':<14} "
                  f"{'Rating':<8} {'CBF':>6} {'NCF':>6}")
            print(f"  {'-'*5} {'-'*45} {'-'*14} {'-'*8} {'-'*6} {'-'*6}")
            for rank, row in results.iterrows():
                ncf_str = f"{row['ncf_score']:.3f}" \
                          if not pd.isna(row.get("ncf_score", np.nan)) else "  N/A"
                print(f"  {rank:<5} {str(row['Course_Name']):<45} "
                      f"{str(row.get('Difficulty','')):<14} "
                      f"{str(row.get('Ratings','')):<8} "
                      f"{row['cbf_score']:>6.3f} {ncf_str:>6}")
        return results


# ── 3. Instantiate recommender ────────────────────────────
print("⚙️  Initialising Hybrid Recommender …")
recommender = HybridRecommender(
    courses        = courses,
    cosine_sim     = cosine_sim,
    course_indices = course_indices,
    ncf_model      = ncf_model,
    user_enc       = user_enc,
    course_enc     = course_enc,
    top_n          = TOP_N,
    cbf_candidates = CBF_CANDIDATES,
)
print("  ✅ Recommender ready\n")


# ── 4. Demo recommendations ───────────────────────────────
print("🔍 Demo Recommendations")
print("─" * 55)

# Demo 1: known user + known course
print("\n[Demo 1] Known user, known course:")
r1 = recommender.recommend(
    course_name = "google cybersecurity",
    user_id     = "U0001"
)

# Demo 2: different user + different course
print("\n[Demo 2] Different user + course:")
r2 = recommender.recommend(
    course_name = "ibm data science",
    user_id     = "U0050"
)

# Demo 3: content-only (no user ID)
print("\n[Demo 3] Content-only (no user ID):")
r3 = recommender.recommend(
    course_name = "machine learning",
    user_id     = None
)


# ── 5. Save sample output CSV ────────────────────────────
print("\n💾 Saving sample output …")
r1["query_course"] = "google cybersecurity"
r1["user_id"]      = "U0001"
r2["query_course"] = "ibm data science"
r2["user_id"]      = "U0050"
r3["query_course"] = "machine learning"
r3["user_id"]      = "None"

sample_output = pd.concat([r1, r2, r3], ignore_index=True)
sample_path   = os.path.join(OUTPUT_DIR, "hybrid_sample_output.csv")
sample_output.to_csv(sample_path, index=False)
print(f"  ✅ Sample output → {sample_path}")


# ── 6. Save recommender object ────────────────────────────
recommender_path = os.path.join(OUTPUT_DIR, "hybrid_recommender.pkl")
# save without model (model saved separately)
save_obj = {
    "course_indices" : course_indices,
    "top_n"          : TOP_N,
    "cbf_candidates" : CBF_CANDIDATES,
}
with open(recommender_path, "wb") as f:
    pickle.dump(save_obj, f)
print(f"  ✅ Recommender config → {recommender_path}")


# ── 7. Summary ────────────────────────────────────────────
print("\n" + "="*55)
print("  📋 HYBRID MODEL SUMMARY")
print("="*55)
print(f"  Strategy       : CBF retrieval → NCF re-ranking")
print(f"  CBF candidates : top {CBF_CANDIDATES} by cosine similarity")
print(f"  Final output   : top {TOP_N} by NCF predicted rating")
print(f"  Cold-start     : falls back to CBF ranking")
print(f"  Files saved to : {OUTPUT_DIR}")
print("="*55)
print("\n✅ Hybrid model complete!")
print("📌 Next step: Build the Flask/FastAPI backend to serve recommendations.\n")
