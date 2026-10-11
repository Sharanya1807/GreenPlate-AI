
import joblib

model = joblib.load("food_waste_model.pkl")

print("Model type:", type(model))
print("Model details:")
print(model)

if hasattr(model, "feature_names_in_"):
    print("\nExpected feature names:")
    print(list(model.feature_names_in_))

if hasattr(model, "n_features_in_"):
    print("\nNumber of input features:", model.n_features_in_)

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(name, "->", type(step))