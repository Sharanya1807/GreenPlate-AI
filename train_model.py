
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folder
folder = Path(__file__).parent

# Load cleaned data
df = pd.read_csv(folder / "cleaned_train.csv")

print("===== GREENPLATE AI: MODEL TRAINING =====")
print("Dataset shape:", df.shape)

# Target: the value our model will predict
target = "food_waste_kg"

# Separate input features (X) and target (y)
X = df.drop(columns=[target, "ID", "date"], errors="ignore").copy()
y = df[target].copy()

# Remove rows with missing target values, if any
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Identify text and numeric columns
categorical_columns = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumeric features:", numeric_columns)
print("Categorical features:", categorical_columns)

# Prepare numeric features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Process each feature type correctly
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Create the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2
    ))
])

# Keep some records separate to test the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_test))

# Train the model
model.fit(X_train, y_train)

# Predict on unseen validation records
predictions = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
print(f"R-squared (R2): {r2:.3f}")

# Show a few actual versus predicted values
comparison = pd.DataFrame({
    "Actual food waste (kg)": y_test.values[:10],
    "Predicted food waste (kg)": predictions[:10]
})

print("\nActual vs predicted:")
print(comparison.round(2).to_string(index=False))

# Save the trained model
model_path = folder / "food_waste_model.pkl"
joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)