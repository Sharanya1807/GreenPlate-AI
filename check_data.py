
import pandas as pd

df = pd.read_csv("cleaned_train.csv")

print("Day values:")
print(sorted(df["day_of_week"].dropna().unique().tolist()))

print("\nStaff experience values:")
print(df["staff_experience"].dropna().unique().tolist())

print("\nFood category values:")
print(df["waste_category"].dropna().unique().tolist())