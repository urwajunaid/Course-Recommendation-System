"""
============================================================
  TF-IDF Vectorization & Cosine Similarity Matrix
  Phase 3 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Inputs  : ../data/cleaned_courses.csv
Outputs :
  ../data/tfidf_vectorizer.pkl   — fitted TF-IDF vectorizer
  ../data/tfidf_matrix.npz       — sparse TF-IDF feature matrix
  ../data/cosine_sim.npy         — cosine similarity matrix (N x N)
  ../data/course_indices.pkl     — course title → row index mapping

Run: python tfidf_vectorizer.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

# ── 0. Config ─────────────────────────────────────────────
DATA_PATH   = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv"
OUTPUT_DIR  = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load cleaned data ──────────────────────────────────
print("\n📂 Loading cleaned dataset …")
df = pd.read_csv(DATA_PATH)

# drop stray index col if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ensure combined_text exists
if "combined_text" not in df.columns:
    raise ValueError(
        "❌ 'combined_text' column missing. "
        "Run data_cleaning.py first."
    )

df = df.dropna(subset=["combined_text"]).reset_index(drop=True)
print(f"  Rows loaded : {len(df)}")
print(f"  Columns     : {df.columns.tolist()}\n")


# ── 2. Text preprocessing helper ─────────────────────────
import re
import nltk

# download quietly
nltk.download("stopwords",      quiet=True)
nltk.download("punkt",          quiet=True)
nltk.download("wordnet",        quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Lowercase → remove punctuation → remove stopwords → lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

print("🔧 Preprocessing combined_text …")
df["processed_text"] = df["combined_text"].apply(clean_text)
print(f"  Sample input  : {df['combined_text'].iloc[0][:80]}…")
print(f"  Sample output : {df['processed_text'].iloc[0][:80]}…\n")


# ── 3. TF-IDF Vectorization ───────────────────────────────
print("📐 Fitting TF-IDF vectorizer …")
tfidf = TfidfVectorizer(
    max_features  = 5000,      # top 5000 terms
    ngram_range   = (1, 2),    # unigrams + bigrams
    min_df        = 2,         # ignore terms in < 2 docs
    max_df        = 0.85,      # ignore terms in > 85% of docs
    sublinear_tf  = True,      # apply log normalization to TF
    strip_accents = "unicode",
    analyzer      = "word",
)

tfidf_matrix = tfidf.fit_transform(df["processed_text"])
print(f"  Vocabulary size   : {len(tfidf.vocabulary_)}")
print(f"  TF-IDF matrix     : {tfidf_matrix.shape}  "
      f"(courses × terms)")
print(f"  Matrix sparsity   : "
      f"{100 * (1 - tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)):.1f}%\n")


# ── 4. Cosine Similarity Matrix ───────────────────────────
print("📏 Computing Cosine Similarity matrix …")
print("  (this may take a few seconds for 1000 courses)")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"  Similarity matrix : {cosine_sim.shape}  "
      f"(courses × courses)\n")

# quick sanity check — diagonal should be 1.0
assert np.allclose(np.diag(cosine_sim), 1.0), \
    "❌ Diagonal values not 1 — something is wrong!"
print("  ✅ Sanity check passed (diagonal = 1.0)\n")


# ── 5. Encode categorical features ───────────────────────
#       (needed later by the NCF model)
print("🔢 Encoding categorical features …")
le_diff = LabelEncoder()
le_cert = LabelEncoder()

df["difficulty_encoded"]    = le_diff.fit_transform(
    df["Difficulty"].fillna("Unknown"))
df["certificate_encoded"]   = le_cert.fit_transform(
    df["Type_Of_Certificate"].fillna("Unknown"))

print(f"  Difficulty classes  : {list(le_diff.classes_)}")
print(f"  Certificate classes : {list(le_cert.classes_)}\n")


# ── 6. Course index mapping ───────────────────────────────
#       Maps Course_Name → row index for fast lookup
course_indices = pd.Series(
    df.index, index=df["Course_Name"].str.lower().str.strip()
).drop_duplicates()


# ── 7. Save all artefacts ─────────────────────────────────
print("💾 Saving artefacts …")

# 7a. TF-IDF vectorizer (pickle)
vectorizer_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(tfidf, f)
print(f"  ✅ Vectorizer  → {vectorizer_path}")

# 7b. Sparse TF-IDF matrix (npz — much smaller than dense)
matrix_path = os.path.join(OUTPUT_DIR, "tfidf_matrix.npz")
sp.save_npz(matrix_path, tfidf_matrix)
print(f"  ✅ TF-IDF matrix → {matrix_path}")

# 7c. Cosine similarity matrix (npy)
sim_path = os.path.join(OUTPUT_DIR, "cosine_sim.npy")
np.save(sim_path, cosine_sim)
print(f"  ✅ Cosine sim  → {sim_path}")

# 7d. Course index mapping (pickle)
idx_path = os.path.join(OUTPUT_DIR, "course_indices.pkl")
with open(idx_path, "wb") as f:
    pickle.dump(course_indices, f)
print(f"  ✅ Course index → {idx_path}")

# 7e. Save df with encoded columns back to CSV
encoded_path = os.path.join(OUTPUT_DIR, "cleaned_courses.csv")
df.to_csv(encoded_path, index=False)
print(f"  ✅ Updated CSV  → {encoded_path}  "
      f"(added processed_text, encoded columns)\n")


# ── 8. Demo — top 5 similar courses for a sample ─────────
print("🔍 Demo: Top 5 courses similar to the first course …")
sample_idx   = 0
sample_name  = df["Course_Name"].iloc[sample_idx]
sim_scores   = list(enumerate(cosine_sim[sample_idx]))
sim_scores   = sorted(sim_scores, key=lambda x: x[1], reverse=True)
top5         = sim_scores[1:6]           # skip index 0 (itself)

print(f"\n  Query course : '{sample_name}'\n")
print(f"  {'Rank':<5} {'Course':<50} {'Similarity':>10}")
print(f"  {'-'*5} {'-'*50} {'-'*10}")
for rank, (idx, score) in enumerate(top5, 1):
    name = df["Course_Name"].iloc[idx]
    print(f"  {rank:<5} {name:<50} {score:>10.4f}")


# ── 9. Summary ────────────────────────────────────────────
print("\n" + "="*55)
print("  📋 TFIDF PIPELINE SUMMARY")
print("="*55)
print(f"  Courses processed     : {len(df)}")
print(f"  Vocabulary size       : {len(tfidf.vocabulary_)}")
print(f"  TF-IDF matrix shape   : {tfidf_matrix.shape}")
print(f"  Cosine sim shape      : {cosine_sim.shape}")
print(f"  Artefacts saved to    : {os.path.abspath(OUTPUT_DIR)}/")
print("="*55)
print("\n✅ TF-IDF & Cosine Similarity pipeline complete!\n")
