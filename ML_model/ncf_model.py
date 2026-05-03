"""
============================================================
  Neural Collaborative Filtering (NCF) Model
  Phase 4 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Inputs  :
  ncf_interactions.csv       — user-course interaction log
  cleaned_courses.csv        — course metadata

Outputs :
  ncf_model.keras            — trained NCF model
  ncf_user_encoder.pkl       — user ID label encoder
  ncf_course_encoder.pkl     — course ID label encoder
  ncf_training_history.png   — loss & MAE curves

Run: python ncf_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── 0. Config ─────────────────────────────────────────────
NCF_PATH    = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\ncf_interactions.csv"
DATA_PATH   = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv"
OUTPUT_DIR  = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data"

# Model hyperparameters
EMBEDDING_DIM   = 32       # size of user/course embedding vectors
MLP_LAYERS      = [128, 64, 32]  # hidden layer sizes
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 0.001
BATCH_SIZE      = 64
EPOCHS          = 30
VALIDATION_SPLIT= 0.15
RANDOM_SEED     = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*55)
print("  NCF MODEL — Neural Collaborative Filtering")
print("="*55)


# ── 1. Load data ──────────────────────────────────────────
print("\n📂 Loading interaction data …")
interactions = pd.read_csv(NCF_PATH)
courses      = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in interactions.columns:
    interactions.drop(columns=["Unnamed: 0"], inplace=True)
if "Unnamed: 0" in courses.columns:
    courses.drop(columns=["Unnamed: 0"], inplace=True)

print(f"  Interactions : {len(interactions)} rows")
print(f"  Courses      : {len(courses)} rows")
print(f"  Columns      : {interactions.columns.tolist()}\n")


# ── 2. Encode user and course IDs ─────────────────────────
print("🔢 Encoding user and course IDs …")
user_enc   = LabelEncoder()
course_enc = LabelEncoder()

interactions["user_idx"]   = user_enc.fit_transform(interactions["user_id"])
interactions["course_idx"] = course_enc.fit_transform(interactions["course_id"])

n_users   = interactions["user_idx"].nunique()
n_courses = interactions["course_idx"].nunique()

print(f"  Unique users   : {n_users}")
print(f"  Unique courses : {n_courses}\n")

# normalize ratings to [0, 1] for sigmoid output
R_MIN, R_MAX = 1.0, 5.0
interactions["rating_norm"] = (interactions["rating"] - R_MIN) / (R_MAX - R_MIN)


# ── 3. Train / test split ─────────────────────────────────
print("✂️  Splitting data …")
X = interactions[["user_idx", "course_idx"]].values
y = interactions["rating_norm"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"  Train : {len(X_train)} | Test : {len(X_test)}\n")


# ── 4. Build NCF Model ────────────────────────────────────
#
#   Architecture: Generalised Matrix Factorisation (GMF)
#                 + MLP branch → concatenated → output
#
#        User ID ──► User Embedding ──┐ GMF branch
#      Course ID ──► Course Embedding─┘ (element-wise product)
#                                     │
#        User ID ──► User Embedding ──┐ MLP branch
#      Course ID ──► Course Embedding─┘ (concatenate → Dense layers)
#                                     │
#                          Concat GMF + MLP → Dense(1, sigmoid) → rating
#
print("🏗️  Building NCF model …")

# ── Inputs
user_input   = keras.Input(shape=(1,), name="user_input")
course_input = keras.Input(shape=(1,), name="course_input")

# ── GMF branch (Matrix Factorisation)
gmf_user_emb   = layers.Embedding(n_users,   EMBEDDING_DIM,
                                   name="gmf_user_emb")(user_input)
gmf_course_emb = layers.Embedding(n_courses, EMBEDDING_DIM,
                                   name="gmf_course_emb")(course_input)
gmf_user_flat   = layers.Flatten()(gmf_user_emb)
gmf_course_flat = layers.Flatten()(gmf_course_emb)
gmf_output      = layers.Multiply()([gmf_user_flat, gmf_course_flat])

# ── MLP branch
mlp_user_emb   = layers.Embedding(n_users,   EMBEDDING_DIM,
                                   name="mlp_user_emb")(user_input)
mlp_course_emb = layers.Embedding(n_courses, EMBEDDING_DIM,
                                   name="mlp_course_emb")(course_input)
mlp_user_flat   = layers.Flatten()(mlp_user_emb)
mlp_course_flat = layers.Flatten()(mlp_course_emb)
mlp_concat      = layers.Concatenate()([mlp_user_flat, mlp_course_flat])

mlp_out = mlp_concat
for units in MLP_LAYERS:
    mlp_out = layers.Dense(units, activation="relu")(mlp_out)
    mlp_out = layers.Dropout(DROPOUT_RATE)(mlp_out)

# ── Merge GMF + MLP
merged = layers.Concatenate()([gmf_output, mlp_out])
output = layers.Dense(1, activation="sigmoid", name="output")(merged)

model = keras.Model(
    inputs  = [user_input, course_input],
    outputs = output,
    name    = "NCF_GMF_MLP"
)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss      = "mse",
    metrics   = ["mae"]
)

model.summary()


# ── 5. Callbacks ──────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor   = "val_loss",
        patience  = 5,
        restore_best_weights = True,
        verbose   = 1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 3,
        verbose  = 1
    )
]


# ── 6. Train ──────────────────────────────────────────────
print("\n🚀 Training NCF model …\n")
history = model.fit(
    x               = [X_train[:, 0], X_train[:, 1]],
    y               = y_train,
    batch_size      = BATCH_SIZE,
    epochs          = EPOCHS,
    validation_split= VALIDATION_SPLIT,
    callbacks       = callbacks,
    verbose         = 1
)


# ── 7. Evaluate ───────────────────────────────────────────
print("\n📊 Evaluating on test set …")
y_pred_norm = model.predict(
    [X_test[:, 0], X_test[:, 1]], verbose=0
).flatten()

# denormalize back to 1–5 scale
y_pred = y_pred_norm * (R_MAX - R_MIN) + R_MIN
y_true = y_test     * (R_MAX - R_MIN) + R_MIN

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\n  ✅ Test MAE  : {mae:.4f}")
print(f"  ✅ Test RMSE : {rmse:.4f}")
print(f"  (Lower is better — MAE < 0.5 is good for 1–5 scale)\n")


# ── 8. Plot training history ──────────────────────────────
print("📈 Saving training history plot …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("NCF Model Training History", fontsize=14, fontweight="bold")

# Loss
axes[0].plot(history.history["loss"],     label="Train Loss", color="#2563EB")
axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#DC2626",
             linestyle="--")
axes[0].set_title("Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()

# MAE
axes[1].plot(history.history["mae"],     label="Train MAE", color="#059669")
axes[1].plot(history.history["val_mae"], label="Val MAE",   color="#F59E0B",
             linestyle="--")
axes[1].set_title("Mean Absolute Error")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "ncf_training_history.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✅ Plot saved → {plot_path}\n")


# ── 9. Save model & encoders ──────────────────────────────
print("💾 Saving model and encoders …")

model_path = os.path.join(OUTPUT_DIR, "ncf_model.keras")
model.save(model_path)
print(f"  ✅ Model         → {model_path}")

user_enc_path = os.path.join(OUTPUT_DIR, "ncf_user_encoder.pkl")
with open(user_enc_path, "wb") as f:
    pickle.dump(user_enc, f)
print(f"  ✅ User encoder  → {user_enc_path}")

course_enc_path = os.path.join(OUTPUT_DIR, "ncf_course_encoder.pkl")
with open(course_enc_path, "wb") as f:
    pickle.dump(course_enc, f)
print(f"  ✅ Course encoder → {course_enc_path}")


# ── 10. Demo — predict top 5 courses for a sample user ───
print("\n🔍 Demo: Top 5 course recommendations for user U0001 …")
sample_user = "U0001"

if sample_user in user_enc.classes_:
    user_idx     = user_enc.transform([sample_user])[0]
    all_course_idx = np.arange(n_courses)
    user_arr     = np.full(n_courses, user_idx)

    preds_norm   = model.predict(
        [user_arr, all_course_idx], verbose=0
    ).flatten()
    preds_rating = preds_norm * (R_MAX - R_MIN) + R_MIN

    # get top 5 course indices
    top5_idx     = np.argsort(preds_rating)[::-1][:5]
    top5_courses = course_enc.inverse_transform(top5_idx)

    print(f"\n  {'Rank':<5} {'Course ID':<12} {'Predicted Rating':>16}")
    print(f"  {'-'*5} {'-'*12} {'-'*16}")
    for rank, (cid, score) in enumerate(
            zip(top5_courses, preds_rating[top5_idx]), 1):
        # try to match course name
        match = courses[courses.index == cid]
        name  = match["Course_Name"].values[0] \
                if len(match) > 0 else f"Course {cid}"
        print(f"  {rank:<5} {str(cid):<12} {score:>14.3f}")
else:
    print(f"  ⚠️ User {sample_user} not found in encoder.")


# ── 11. Summary ───────────────────────────────────────────
print("\n" + "="*55)
print("  📋 NCF MODEL SUMMARY")
print("="*55)
print(f"  Users              : {n_users}")
print(f"  Courses            : {n_courses}")
print(f"  Embedding dim      : {EMBEDDING_DIM}")
print(f"  MLP layers         : {MLP_LAYERS}")
print(f"  Epochs trained     : {len(history.history['loss'])}")
print(f"  Test MAE           : {mae:.4f}")
print(f"  Test RMSE          : {rmse:.4f}")
print(f"  Model saved to     : {OUTPUT_DIR}")
print("="*55)
print("\n✅ NCF model training complete!")
print("📌 Next step: Build the Hybrid model combining NCF + Content-Based.\n")
