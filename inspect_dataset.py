
import pandas as pd
from pathlib import Path

# Locate train.csv in this folder
folder = Path(__file__).parent
file_path = folder / "train.csv"

print("GreenPlate AI - Dataset Inspection")

if not file_path.exists():
    print("ERROR: train.csv not found!")
else:
    df = pd.read_csv(file_path)

    print("\nDataset size (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes.to_string())

    report_path = folder / "dataset_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATASET SIZE\n")
        f.write(str(df.shape) + "\n\n")
        f.write("COLUMN NAMES\n")
        f.write(str(df.columns.tolist()) + "\n\n")
        f.write("FIRST FIVE ROWS\n")
        f.write(df.head().to_string() + "\n\n")
        f.write("MISSING VALUES\n")
        f.write(df.isnull().sum().to_string() + "\n")

    print("\nInspection complete!")
    print("Report saved to:", report_path)