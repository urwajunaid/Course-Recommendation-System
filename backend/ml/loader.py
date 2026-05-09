"""
ml/loader.py
Loads all ML artefacts once at startup and exposes a singleton.
"""

import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import settings

# Lazy-loaded singletons
_state = {
    "loaded"        : False,
    "courses"       : None,
    "cosine_sim"    : None,
    "course_indices": None,
    "ncf_model"     : None,
    "user_enc"      : None,
    "course_enc"    : None,
    "tfidf_vec"     : None,
    "recommender"   : None,
}


def _abs_data(filename: str) -> str:
    """Resolve DATA_DIR (which may be relative to backend/) to an absolute path."""
    data_dir = settings.DATA_DIR
    if not os.path.isabs(data_dir):
        # Resolve relative to the directory containing this file (backend/ml/)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.normpath(os.path.join(base, data_dir))
    return os.path.join(data_dir, filename)


def load_models():
    """Called once at FastAPI startup."""
    if _state["loaded"]:
        return

    import pandas as pd
    from tensorflow import keras

    data_dir = _abs_data("")
    print(f"\n[INFO] Loading ML artefacts from: {data_dir}")

    # ── Courses CSV ───────────────────────────────────────
    csv_path = _abs_data("cleaned_courses.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"cleaned_courses.csv not found at: {csv_path}")
    _state["courses"] = pd.read_csv(csv_path)
    if "Unnamed: 0" in _state["courses"].columns:
        _state["courses"].drop(columns=["Unnamed: 0"], inplace=True)
    _state["courses"] = _state["courses"].reset_index(drop=True)
    print(f"  [OK] Courses: {len(_state['courses'])} rows")

    # ── Cosine similarity matrix ──────────────────────────
    sim_path = _abs_data("cosine_sim.npy")
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"cosine_sim.npy not found at: {sim_path}")
    _state["cosine_sim"] = np.load(sim_path)
    print(f"  [OK] Cosine sim: {_state['cosine_sim'].shape}")

    # ── Course indices (Pandas Series: lowercased name → row index) ──
    idx_path = _abs_data("course_indices.pkl")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"course_indices.pkl not found at: {idx_path}")
    with open(idx_path, "rb") as f:
        _state["course_indices"] = pickle.load(f)
    print(f"  [OK] Course indices: {len(_state['course_indices'])} entries")

    # ── TF-IDF vectorizer (optional) ─────────────────────
    tfidf_path = _abs_data("tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            _state["tfidf_vec"] = pickle.load(f)
        print("  [OK] TF-IDF vectorizer loaded")

    # ── NCF model ─────────────────────────────────────────
    model_path = _abs_data("ncf_model.keras")
    if os.path.exists(model_path):
        _state["ncf_model"] = keras.models.load_model(model_path)
        print("  [OK] NCF model loaded")
    else:
        print("  [WARN] ncf_model.keras not found — NCF re-ranking disabled")

    # ── Encoders ──────────────────────────────────────────
    user_enc_path   = _abs_data("ncf_user_encoder.pkl")
    course_enc_path = _abs_data("ncf_course_encoder.pkl")

    if os.path.exists(user_enc_path):
        with open(user_enc_path, "rb") as f:
            _state["user_enc"] = pickle.load(f)
        print("  [OK] User encoder loaded")
    else:
        print("  [WARN] ncf_user_encoder.pkl not found")

    if os.path.exists(course_enc_path):
        with open(course_enc_path, "rb") as f:
            _state["course_enc"] = pickle.load(f)
        print("  [OK] Course encoder loaded")
    else:
        print("  [WARN] ncf_course_encoder.pkl not found")

    # ── Build HybridRecommender ───────────────────────────
    from ml.recommender import HybridRecommender
    _state["recommender"] = HybridRecommender(
        courses        = _state["courses"],
        cosine_sim     = _state["cosine_sim"],
        course_indices = _state["course_indices"],
        ncf_model      = _state["ncf_model"],
        user_enc       = _state["user_enc"],
        course_enc     = _state["course_enc"],
        top_n          = 10,
        cbf_candidates = 50,
    )
    _state["loaded"] = True
    print("  [OK] HybridRecommender ready\n")


def get_recommender():
    if not _state["loaded"]:
        load_models()
    return _state["recommender"]


def get_courses_df():
    return _state["courses"]
