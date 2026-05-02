"""
============================================================
  EDA — Hybrid Course Recommendation System
  Phase 2 | NUCES FAST | Data & Evaluation: Urwa Junaid
============================================================
Run: python eda.py
Output: saves all plots to ./eda_plots/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings("ignore")

# ── 0. Config ────────────────────────────────────────────
DATA_PATH  = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv"   # adjust if needed
OUTPUT_DIR = r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "primary"   : "#2563EB",
    "secondary" : "#7C3AED",
    "accent"    : "#059669",
    "warn"      : "#DC2626",
    "bg"        : "#F8FAFC",
    "text"      : "#1E293B",
}
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor" : PALETTE["bg"],
    "axes.facecolor"   : PALETTE["bg"],
    "axes.edgecolor"   : "#CBD5E1",
    "axes.labelcolor"  : PALETTE["text"],
    "xtick.color"      : PALETTE["text"],
    "ytick.color"      : PALETTE["text"],
    "text.color"       : PALETTE["text"],
    "grid.color"       : "#E2E8F0",
    "grid.linewidth"   : 0.6,
})

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✅ Saved → {path}")

# ── 1. Load ──────────────────────────────────────────────
print("\n📂 Loading dataset …")
df = pd.read_csv(DATA_PATH)

# clean up stray index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# strip "Skills you'll gain: " prefix if not already done
df["Skills"] = df["Skills"].str.replace(
    r"(?i)skills you'?ll gain:\s*", "", regex=True
).str.strip()

# clean Reviews → numeric
df["Reviews_Num"] = (
    df["Reviews"]
    .str.replace(r"[^\d.]", "", regex=True)
    .replace("", np.nan)
    .astype(float)
)
# 'K' multiplier was stripped, restore thousands
# raw looks like "19K reviews" → after strip → "19" (lost K)
# re-extract properly
df["Reviews_Num"] = df["Reviews"].apply(
    lambda x: (
        float(re.search(r"[\d.]+", str(x)).group()) * 1000
        if "K" in str(x).upper() and re.search(r"[\d.]+", str(x))
        else (float(re.search(r"[\d.]+", str(x)).group())
              if re.search(r"[\d.]+", str(x)) else np.nan)
    )
)

print(f"  Shape        : {df.shape}")
print(f"  Columns      : {df.columns.tolist()}")
print(f"\n🔹 Missing Values:\n{df.isnull().sum()}")
print(f"\n🔹 Basic Stats:\n{df.describe(include='all').T}\n")


# ═══════════════════════════════════════════════════════════
# PLOT 1 — Ratings Distribution
# ═══════════════════════════════════════════════════════════
print("📊 Plot 1: Ratings Distribution")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Course Ratings Distribution", fontsize=15, fontweight="bold",
             color=PALETTE["text"], y=1.01)

# histogram
axes[0].hist(df["Ratings"].dropna(), bins=25, color=PALETTE["primary"],
             edgecolor="white", linewidth=0.6)
axes[0].set_title("Histogram of Ratings", fontsize=12)
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")
med = df["Ratings"].median()
axes[0].axvline(med, color=PALETTE["warn"], linestyle="--", linewidth=1.5,
                label=f"Median = {med}")
axes[0].legend(fontsize=9)

# boxplot by difficulty
order = [d for d in df["Difficulty"].unique()
         if d not in ["not found", "Not Found"]]
sub = df[df["Difficulty"].isin(order)]
colors = [PALETTE["accent"], PALETTE["primary"],
          PALETTE["secondary"], "#F59E0B"]
bp = axes[1].boxplot(
    [sub[sub["Difficulty"] == d]["Ratings"].dropna() for d in order],
    labels=order, patch_artist=True, medianprops=dict(color="white", linewidth=2)
)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.8)
axes[1].set_title("Ratings by Difficulty", fontsize=12)
axes[1].set_xlabel("Difficulty Level")
axes[1].set_ylabel("Rating")

plt.tight_layout()
save(fig, "01_ratings_distribution.png")


# ═══════════════════════════════════════════════════════════
# PLOT 2 — Difficulty Level Distribution
# ═══════════════════════════════════════════════════════════
print("📊 Plot 2: Difficulty Distribution")
diff_counts = df["Difficulty"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(diff_counts.index, diff_counts.values,
              color=[PALETTE["accent"], PALETTE["primary"],
                     PALETTE["secondary"], "#F59E0B", "#94A3B8"][:len(diff_counts)],
              edgecolor="white", linewidth=0.8)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Difficulty Level Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Number of Courses")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
save(fig, "02_difficulty_distribution.png")


# ═══════════════════════════════════════════════════════════
# PLOT 3 — Certificate Type Distribution
# ═══════════════════════════════════════════════════════════
print("📊 Plot 3: Certificate Type")
cert_counts = df["Type_Of_Certificate"].value_counts()
colors_pie = [PALETTE["primary"], PALETTE["secondary"],
              PALETTE["accent"], "#F59E0B", "#94A3B8"]
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    cert_counts.values, labels=cert_counts.index,
    autopct="%1.1f%%", colors=colors_pie[:len(cert_counts)],
    startangle=140, pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("white")
    at.set_fontweight("bold")
ax.set_title("Distribution by Certificate Type",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "03_certificate_type.png")


# ═══════════════════════════════════════════════════════════
# PLOT 4 — Duration Distribution
# ═══════════════════════════════════════════════════════════
print("📊 Plot 4: Duration Distribution")
dur_counts = df["Duration"].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(dur_counts.index, dur_counts.values,
               color=PALETTE["secondary"], edgecolor="white",
               linewidth=0.8, height=0.6)
for bar in bars:
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=10,
            fontweight="bold")
ax.set_title("Course Duration Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Courses")
ax.invert_yaxis()
plt.tight_layout()
save(fig, "04_duration_distribution.png")


# ═══════════════════════════════════════════════════════════
# PLOT 5 — Top 15 Companies / Providers
# ═══════════════════════════════════════════════════════════
print("📊 Plot 5: Top 15 Companies")
top_companies = df["Company_Name"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(11, 6))
colors_grad = sns.color_palette("Blues_r", len(top_companies))
bars = ax.barh(top_companies.index[::-1], top_companies.values[::-1],
               color=colors_grad[::-1], edgecolor="white", linewidth=0.6)
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=9,
            fontweight="bold", color=PALETTE["text"])
ax.set_title("Top 15 Course Providers", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Courses")
plt.tight_layout()
save(fig, "05_top_companies.png")


# ═══════════════════════════════════════════════════════════
# PLOT 6 — Top 25 Most Common Skills
# ═══════════════════════════════════════════════════════════
print("📊 Plot 6: Top 25 Skills")
all_skills = []
for entry in df["Skills"].dropna():
    all_skills.extend([s.strip().title() for s in entry.split(",")])
skill_counts = Counter(all_skills)
top_skills = pd.DataFrame(skill_counts.most_common(25),
                           columns=["Skill", "Count"])

fig, ax = plt.subplots(figsize=(12, 7))
colors_grad2 = sns.color_palette("viridis", len(top_skills))
bars = ax.barh(top_skills["Skill"][::-1], top_skills["Count"][::-1],
               color=colors_grad2, edgecolor="white", linewidth=0.5)
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=8,
            fontweight="bold")
ax.set_title("Top 25 Most Common Skills Across Courses",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Frequency")
plt.tight_layout()
save(fig, "06_top_skills.png")


# ═══════════════════════════════════════════════════════════
# PLOT 7 — Ratings vs Reviews (scatter)
# ═══════════════════════════════════════════════════════════
print("📊 Plot 7: Ratings vs Reviews")
plot_df = df.dropna(subset=["Ratings", "Reviews_Num"])
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    plot_df["Reviews_Num"], plot_df["Ratings"],
    c=plot_df["Ratings"], cmap="coolwarm",
    alpha=0.6, edgecolors="white", linewidths=0.4, s=50
)
plt.colorbar(scatter, ax=ax, label="Rating")
ax.set_xscale("log")
ax.set_title("Course Ratings vs. Number of Reviews",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Reviews (log scale)")
ax.set_ylabel("Rating")
plt.tight_layout()
save(fig, "07_ratings_vs_reviews.png")


# ═══════════════════════════════════════════════════════════
# PLOT 8 — Heatmap: Difficulty × Certificate Type
# ═══════════════════════════════════════════════════════════
print("📊 Plot 8: Difficulty × Certificate Type Heatmap")
pivot = pd.crosstab(df["Difficulty"], df["Type_Of_Certificate"])
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Course Count"}, ax=ax)
ax.set_title("Difficulty Level vs. Certificate Type",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Certificate Type")
ax.set_ylabel("Difficulty Level")
plt.tight_layout()
save(fig, "08_difficulty_vs_certificate_heatmap.png")


# ═══════════════════════════════════════════════════════════
# PLOT 9 — Skills per Course Distribution
# ═══════════════════════════════════════════════════════════
print("📊 Plot 9: Skills per Course")
df["Skill_Count"] = df["Skills"].dropna().apply(
    lambda x: len([s for s in x.split(",") if s.strip()])
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["Skill_Count"].dropna(), bins=20,
        color=PALETTE["accent"], edgecolor="white", linewidth=0.7)
mean_skills = df["Skill_Count"].mean()
ax.axvline(mean_skills, color=PALETTE["warn"], linestyle="--",
           linewidth=1.5, label=f"Mean = {mean_skills:.1f}")
ax.set_title("Number of Skills Listed per Course",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Skill Count")
ax.set_ylabel("Number of Courses")
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "09_skills_per_course.png")


# ═══════════════════════════════════════════════════════════
# PLOT 10 — Combined Text Length Distribution
#            (proxy for description richness → TF-IDF input)
# ═══════════════════════════════════════════════════════════
print("📊 Plot 10: Combined Text Length")
if "combined_text" in df.columns:
    df["text_len"] = df["combined_text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["text_len"].dropna(), bins=30,
            color=PALETTE["secondary"], edgecolor="white", linewidth=0.7)
    ax.axvline(df["text_len"].median(), color=PALETTE["warn"],
               linestyle="--", linewidth=1.5,
               label=f"Median = {df['text_len'].median():.0f} words")
    ax.set_title("Combined Text Length Distribution (TF-IDF Input)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Courses")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "10_combined_text_length.png")
else:
    print("  ⚠️  'combined_text' column not found — skipping Plot 10.")
    print("     Run the cleaning script first to generate combined_text.")


# ── Summary ──────────────────────────────────────────────
print("\n" + "="*55)
print("  📋 EDA SUMMARY")
print("="*55)
print(f"  Total courses       : {len(df)}")
print(f"  Unique providers    : {df['Company_Name'].nunique()}")
print(f"  Avg rating          : {df['Ratings'].mean():.2f}  (±{df['Ratings'].std():.2f})")
print(f"  Most common diff.   : {df['Difficulty'].mode()[0]}")
print(f"  Most common cert.   : {df['Type_Of_Certificate'].mode()[0]}")
print(f"  Top provider        : {df['Company_Name'].value_counts().idxmax()}")
print(f"  Top skill           : {top_skills.iloc[0]['Skill']}")
print(f"  Avg skills/course   : {df['Skill_Count'].mean():.1f}")
print(f"\n  Plots saved to      : {os.path.abspath(OUTPUT_DIR)}/")
print("="*55)
print("\n✅ EDA complete!\n")