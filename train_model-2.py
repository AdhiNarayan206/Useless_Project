import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load synthetic data
df = pd.read_csv("lays_synthetic_data-updated.csv")  # Updated to match our latest file

# Encode flavor
le = LabelEncoder()
df["flavor"] = le.fit_transform(df["flavor"])

# Features and target
X = df[["weight_g", "flavor"]]
y = df["chip_count"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test prediction
test_packet_df = pd.DataFrame([[10.0, le.transform(["Classic"])[0]]], columns=["weight_g", "flavor"])
prediction = model.predict(test_packet_df)
print(f"Your packet contains... {int(prediction[0])} chips!")

# Evaluate model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Function to predict chips (for API use)
def predict_chips(weight, flavor_str):
    encoded_flavor = le.transform([flavor_str])[0]
    test_data = pd.DataFrame([[weight, encoded_flavor]], columns=["weight_g", "flavor"])
    prediction = model.predict(test_data)
    return int(prediction[0])

# Save model and encoder
joblib.dump(model, "chip_predictor_model.pkl")
joblib.dump(le, "flavor_encoder.pkl")
print("Model and encoder saved as chip_predictor_model.pkl and flavor_encoder.pkl!")