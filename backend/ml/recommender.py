"""
ml/recommender.py
HybridRecommender — CBF retrieval + optional NCF re-ranking.

course_indices  : Pandas Series  {lowercased_name → row_index_in_courses_df}
course_enc      : LabelEncoder   fitted on integer course IDs (0..N)
user_enc        : LabelEncoder   fitted on string user IDs ('U0000'..)
"""

import numpy as np
import pandas as pd

R_MIN, R_MAX = 1.0, 5.0


class HybridRecommender:

    def __init__(self, courses, cosine_sim, course_indices,
                 ncf_model, user_enc, course_enc,
                 top_n=10, cbf_candidates=50):
        self.courses        = courses
        self.cosine_sim     = cosine_sim
        self.course_indices = course_indices   # Series: lowercase name → row idx
        self.ncf_model      = ncf_model
        self.user_enc       = user_enc
        self.course_enc     = course_enc
        self.top_n          = top_n
        self.cbf_candidates = cbf_candidates

    # ── Stage 1: Content-Based Filtering ─────────────────
    def _cbf_candidates(self, course_name: str) -> pd.DataFrame:
        key = course_name.lower().strip()

        if key not in self.course_indices:
            # Fuzzy fallback: partial match
            matches = [c for c in self.course_indices.index
                       if key in c or c in key]
            if not matches:
                raise ValueError(
                    f"Course '{course_name}' not found in dataset. "
                    "Try a shorter or different name."
                )
            key = matches[0]

        # course_indices may map one name to multiple rows (pandas Series)
        raw_idx = self.course_indices[key]
        if hasattr(raw_idx, '__len__'):
            idx = int(raw_idx.iloc[0]) if hasattr(raw_idx, 'iloc') else int(raw_idx[0])
        else:
            idx = int(raw_idx)

        sim_scores = sorted(
            enumerate(self.cosine_sim[idx]),
            key=lambda x: x[1], reverse=True
        )
        # Skip the query course itself (index 0 is always itself)
        sim_scores = sim_scores[1: self.cbf_candidates + 1]

        candidate_indices      = [i for i, _ in sim_scores]
        candidates             = self.courses.iloc[candidate_indices].copy()
        candidates["cbf_score"]    = [s for _, s in sim_scores]
        candidates["original_idx"] = candidate_indices   # row index == course integer ID
        return candidates

    # ── Stage 2: NCF Re-ranking ───────────────────────────
    def _ncf_rerank(self, candidates: pd.DataFrame,
                    user_id: str) -> pd.DataFrame:
        # NCF unavailable → fall back to CBF order
        if self.ncf_model is None or self.user_enc is None or self.course_enc is None:
            candidates["ncf_score"] = np.nan
            return candidates.sort_values("cbf_score", ascending=False).head(self.top_n)

        # Unknown user → cold-start, use CBF order
        if user_id not in self.user_enc.classes_:
            candidates["ncf_score"] = 3.5
            return candidates.sort_values("cbf_score", ascending=False).head(self.top_n)

        user_idx   = self.user_enc.transform([user_id])[0]
        course_ids = candidates["original_idx"].values   # integer course IDs

        valid_rows, valid_encs = [], []
        known_classes = set(self.course_enc.classes_)
        for row_pos, cid in enumerate(course_ids):
            if cid in known_classes:
                enc_idx = self.course_enc.transform([cid])[0]
                valid_rows.append(row_pos)
                valid_encs.append(enc_idx)

        if not valid_encs:
            candidates["ncf_score"] = 3.5
            return candidates.sort_values("cbf_score", ascending=False).head(self.top_n)

        user_arr   = np.full(len(valid_encs), user_idx)
        course_arr = np.array(valid_encs)
        preds_norm = self.ncf_model.predict(
            [user_arr, course_arr], verbose=0
        ).flatten()
        preds = preds_norm * (R_MAX - R_MIN) + R_MIN

        candidates["ncf_score"] = np.nan
        for row_pos, score in zip(valid_rows, preds):
            candidates.iloc[
                row_pos, candidates.columns.get_loc("ncf_score")
            ] = score

        candidates["ncf_score"] = candidates["ncf_score"].fillna(3.5)
        return candidates.sort_values("ncf_score", ascending=False).head(self.top_n)

    # ── Public API ────────────────────────────────────────
    def recommend(self, course_name: str,
                  user_id: str = None,
                  top_n: int = None) -> list[dict]:
        """Returns a list of dicts ready for JSON serialisation."""
        if top_n:
            self.top_n = top_n

        candidates = self._cbf_candidates(course_name)

        if user_id:
            results = self._ncf_rerank(candidates, user_id)
        else:
            results = candidates.sort_values("cbf_score", ascending=False).head(self.top_n)
            results["ncf_score"] = np.nan

        # Pick whichever columns are available
        want = ["Course_Name", "Company_Name", "Difficulty",
                "Ratings", "Duration", "Type_Of_Certificate",
                "Skills", "cbf_score", "ncf_score"]
        output_cols = [c for c in want if c in results.columns]
        results = results[output_cols].reset_index(drop=True)
        results.index += 1

        records = []
        for rank, row in results.iterrows():
            records.append({
                "rank"            : rank,
                "course_name"     : row.get("Course_Name", ""),
                "company_name"    : row.get("Company_Name", ""),
                "difficulty"      : row.get("Difficulty", ""),
                "ratings"         : row.get("Ratings"),
                "duration"        : row.get("Duration", ""),
                "certificate_type": row.get("Type_Of_Certificate", ""),
                "skills"          : row.get("Skills", ""),
                "cbf_score"       : round(float(row["cbf_score"]), 4),
                "ncf_score"       : (
                    None if pd.isna(row.get("ncf_score", float("nan")))
                    else round(float(row["ncf_score"]), 3)
                ),
            })
        return records

    def search_courses(self, query: str, limit: int = 20) -> list[str]:
        """Fast in-memory substring search for autocomplete."""
        q    = query.lower().strip()
        mask = self.courses["Course_Name"].str.lower().str.contains(q, na=False)
        return self.courses[mask]["Course_Name"].head(limit).tolist()
