import streamlit as st
import pandas as pd
import os
import gdown
from joblib import load
from importlib.metadata import version

# --- Load the model from joblib file ---
@st.cache_resource
def load_model():
    model_path = "car_price_model.joblib"
    if not os.path.exists(model_path):
        gdown.download(
            "https://drive.google.com/uc?id=183946cS8Mjmcz7-qiVGU21_84iVDr5ZH",
            model_path,
            quiet=False
        )
    return load(model_path)

model = load_model()

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "mailto:support@example.com"
    }
)

st.title("🚗 Car Price Predictor")
st.caption(f"v2.1 | sklearn v{version('scikit-learn')}")

with st.expander("ℹ️ About this app"):
    st.write("""
    This app predicts used car prices using machine learning.
    For accurate results, please provide complete information.
    """)

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", [
            'Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'Mitsubishi', 'KIA', 'Other Brands',
            'Nissan', 'BMW', 'Mazda', 'Chevrolet', 'Daewoo', 'Hyundai', 'FAW',
            'Mercedes', 'Classic & Antiques', 'Lexus', 'Audi', 'Range Rover', 'Changan',
            'Porsche', 'Subaru', 'Land Rover', 'Others'
        ])
        condition = st.radio("Condition", ['Used', 'New'], horizontal=True)
        year = st.slider("Manufacturing Year", 1990, 2024, 2018)
        model_name = st.text_input("Model Name", placeholder="Corolla Altis, Civic Oriel etc.")

    with col2:
        registered_city = st.selectbox("Registered City", sorted([
            'Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Multan', 'Other'
        ]))
        fuel = st.selectbox("Fuel Type", [
            'Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'
        ])
        kms_driven = st.number_input("KMs Driven", min_value=0, value=50000, step=1000)
        transaction_type = st.selectbox("Transaction Type", ["Cash", "Installment"])
        wanted_price = st.number_input("Your Wanted Price (PKR)", min_value=0, value=1500000, step=50000)

    submit = st.form_submit_button("Predict Price", type="primary")

# --- Prediction Logic ---
if submit:
    if not model_name.strip():
        st.warning("Please enter the car model name")
    else:
        try:
            age = 2025 - year
            price_per_km = wanted_price / kms_driven if kms_driven > 0 else 0

            input_df = pd.DataFrame([{
                'Brand': brand,
                'Condition': condition,
                'Fuel': fuel,
                'KMs Driven': kms_driven,
                'Model': model_name,
                'Registered City': registered_city,
                'Year': year,
                'Price Per KM': price_per_km,
                'Age': age,
                'Transaction Type': transaction_type,
                'Your Wanted Price': wanted_price
            }])

            with st.spinner("🤖 Calculating fair price..."):
                prediction = model.predict(input_df)[0]

            st.balloons()
            st.success(f"### Estimated Value: PKR {prediction:,.0f}")
            with st.expander("Details"):
                st.json(input_df.iloc[0].to_dict())

        except Exception as e:
            st.error("Prediction failed:")
            st.code(str(e), language='python')
