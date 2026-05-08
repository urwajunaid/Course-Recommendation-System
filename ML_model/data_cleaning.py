import pandas as pd

# STEP 1: Load dataset
df = pd.read_csv(r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv")
# STEP 2: Basic info
print("🔹 Dataset Shape:", df.shape)
print("\n🔹 Columns:", df.columns.tolist())
print("\n🔹 First 5 rows:")
print(df.head())

# STEP 3: Check missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# STEP 4: Drop unnecessary columns
# 'Unnamed: 0' is a leftover index; 'Course_Banner' is just a URL (not useful for ML)
columns_to_keep = ['Course_Name', 'Company_Name', 'Skills', 'Ratings', 'Reviews', 'Difficulty', 'Type_Of_Certificate', 'Duration']
df = df[columns_to_keep]

# STEP 5: Handle missing values
# Difficulty, Type_Of_Certificate, Duration have ~23-24 missing rows — safe to drop
df = df.dropna(subset=['Difficulty', 'Type_Of_Certificate', 'Duration'])
# For Ratings/Reviews, fill with median/unknown instead of dropping
df['Ratings'] = df['Ratings'].fillna(df['Ratings'].median())
df['Reviews'] = df['Reviews'].fillna('(0 reviews)')

# Drop "not found" difficulty — too vague to be useful
df = df[~df['Difficulty'].str.lower().str.strip().eq('not found')]

# STEP 6: Remove duplicates
df = df.drop_duplicates()

# STEP 7: Clean up the Skills column
# It has a prefix like "Skills you'll gain: " — strip it
df['Skills'] = df['Skills'].str.replace(r"Skills you'll gain:\s*", "", regex=True)

# STEP 8: Normalize text
df['Course_Name'] = df['Course_Name'].str.lower().str.strip()
df['Skills'] = df['Skills'].str.lower().str.strip()
df['Company_Name'] = df['Company_Name'].str.lower().str.strip()
df['Difficulty'] = df['Difficulty'].str.lower().str.strip()

# STEP 9: Combine features for NLP/recommendation tasks
#df['combined_text'] = df['Course_Name'] + " " + df['Skills'] + " " + df['Company_Name']
# STEP 9: Combine features for NLP/recommendation tasks
df['combined_text'] = (
    df['Course_Name'].fillna("") + " " +
    df['Skills'].fillna("") + " " +
    df['Skills'].fillna("") + " " +       # repeated for higher TF-IDF weight
    df['Skills'].fillna("") + " " +       # repeated for higher TF-IDF weight
    df['Difficulty'].fillna("") + " " +
    df['Type_Of_Certificate'].fillna("") + " " +
    df['Company_Name'].fillna("") + " " +
    df['Duration'].fillna("")
)

# STEP 10: Save cleaned dataset
print("\n🔹 Combined text word count stats:")
print(df['combined_text'].str.split().str.len().describe())

df.to_csv(
    r"C:\Users\Urwa\OneDrive\Desktop\semester 8\RS\Course Recommendation System\data\cleaned_courses.csv",
    index=False
)
print("\n✅ Data cleaning completed!")
print(f"Final shape: {df.shape}")
print("Saved as cleaned_courses.csv")