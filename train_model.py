import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
df =pd.read_csv("lays_synthetic_data-updated.csv")
le=LabelEncoder()
df["flavor"]=le.fit_transform(df["flavor"])
X = df[["weight_g", "flavor",]]
y = df["chip_count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
test_packet_df = pd.DataFrame([[10.0, le.transform(["Classic"])[0]],], columns=["weight_g", "flavor"])
prediction = model.predict(test_packet_df)
print(f"Your packet contains... {int(prediction[0])} chips!")
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
def predict_chips(weight, flavor_str):
    encoded_flavor = le.transform([flavor_str])[0]
    test_data = [[weight, encoded_flavor]]
    test_data = pd.DataFrame([[weight, encoded_flavor]], columns=["weight_g", "flavor"])
    prediction = model.predict(test_data)
    return int(prediction[0])
import joblib as jb
jb.dump(model,"chip_count_predictor.pk1")