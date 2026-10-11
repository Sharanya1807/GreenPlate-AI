
import pandas as pd
from pathlib import Path

# Find the folder containing this Python file
folder = Path(__file__).parent

# Load the original dataset
df = pd.read_csv(folder / "train.csv")

print("===== GREENPLATE AI: DATA CLEANING =====")
print("Original dataset shape:", df.shape)

# 1. Standardize text columns
for column in ["staff_experience", "waste_category"]:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

# 2. Handle missing staff experience
# Fill missing entries with the most common category
most_common = df["staff_experience"].mode(dropna=True)

if not most_common.empty:
    df["staff_experience"] = df["staff_experience"].fillna(
        most_common.iloc[0]
    )

# 3. Convert the date column to a date format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 4. Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 5. Check target values
print("\nFood waste summary (kg):")
print(df["food_waste_kg"].describe())

# 6. Save the cleaned dataset separately
output_path = folder / "cleaned_train.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:")
print(output_path)

print("\nFinal dataset shape:", df.shape)
print("Cleaning complete!")