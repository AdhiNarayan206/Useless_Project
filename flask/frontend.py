import streamlit as st
import requests
import plotly.express as px
import pandas as pd

st.title("CrispProphet 🥔")
st.write("Predict the number of chips in your Lay's packet with *useless* precision!")

# Input form
flavor = st.selectbox("Select Flavor", pd.read_csv("lays_synthetic_data-updated.csv")["flavor"].unique())
weight = st.number_input("Packet Weight (grams)", min_value=10.0, max_value=60.0, value=11.0)

# Prediction button
if st.button("Predict My Chips!"):
    # Prepare data for API call
    data = {
        "flavor": flavor,
        "weight_g": weight
    }
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json=data,
            headers={"Content-Type": "application/json"},  # Explicitly set header
            timeout=10
        )
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        st.markdown(f"**{result['prediction']}**")
        st.write(result['pun'])
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {str(e)}")

# 2D Plot
df = pd.read_csv("lays_synthetic_data-updated.csv")
fig = px.scatter(df, x="weight_g", y="chip_count", color="flavor", title="Weight vs Chip Count")
st.plotly_chart(fig)

# Fancy button CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #FF4500;
    }
    </style>
    """, unsafe_allow_html=True)