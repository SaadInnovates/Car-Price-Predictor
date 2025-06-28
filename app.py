import streamlit as st
import pandas as pd
from joblib import load
import os
import gdown

# --- Model Loading with Joblib ---
@st.cache_resource(ttl=24*3600)  # Cache for 24 hours
def load_model():
    model_path = "car_price_model.joblib"
    file_id = "183946cS8Mjmcz7-qiVGU21_84iVDr5ZH"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive (this may take a minute)..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=True)
    
    try:
        return load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Please ensure:")
        st.error("1. You have scikit-learn installed")
        st.error("2. The file exists in Google Drive")
        st.stop()

model = load_model()

# --- App UI ---
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")
st.markdown("Enter your car's details to get an estimated resale price.")

brands = [
    'Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'Mitsubishi', 'KIA', 'Other Brands',
    'Nissan', 'BMW', 'Mazda', 'Chevrolet', 'Daewoo', 'Hyundai', 'FAW',
    'Mercedes', 'Classic & Antiques', 'Lexus', 'Audi', 'Range Rover', 'Changan',
    'Porsche', 'Subaru', 'Land Rover', 'Others'
]

conditions = ['Used', 'New']
fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other']
registered_cities = sorted([
    'Karachi', 'Hyderabad', 'Bagh', 'Sukkar', 'Bahawalnagar', 'Lahore',
    'Askoley', 'Khanpur', 'Quetta', 'Karak', 'Islamabad', 'Sialkot',
    'Pakpattan', 'Lasbela', 'Sukkur', 'Rawalpindi', 'Bahawalpur',
    'Ali Masjid', 'Multan', 'Khaplu', 'Tank', 'Badin',
    'Rahimyar Khan', 'Chilas', 'Kasur', 'Khushab', 'Vehari', 'Chitral',
    'Khanewal', 'Attock', 'Larkana', 'Bela', 'Khairpur', 'Kandhura',
    'Abbottabad', 'Nawabshah', 'Bhimber', 'Mardan', 'Chiniot',
    'Faisalabad', 'Sahiwal', 'Haripur', 'Peshawar', 'Kohat',
    'Sargodha', 'Jhelum', 'Gujrat', 'Nowshera', 'Gujranwala', 'Mirpur',
    'Burewala', 'Mandi Bahauddin', 'Muzaffargarh', 'Wah',
    'Dera Ghazi Khan', 'Sheikhupura', 'Okara', 'Dera Ismail Khan',
    'Swat', 'Swabi', 'Muzaffarabad', 'Other'
])
transaction_types = ['Cash', 'Installment/Leasing', 'Other']

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", brands)
        condition = st.selectbox("Condition", conditions)
        fuel = st.selectbox("Fuel Type", fuel_types)
        year = st.slider("Year", 1990, 2024, 2018)

    with col2:
        registered_city = st.selectbox("Registered City", registered_cities)
        transaction = st.selectbox("Transaction Type", transaction_types)
        kms_driven = st.number_input("KMs Driven", min_value=0, step=1000)
        asking_price = st.number_input("Your Asking Price (optional)", min_value=0)

    model_name = st.text_input("Model (e.g., Corolla Altis, Civic Oriel)")

    submit = st.form_submit_button("Predict Price")

if submit:
    if model_name.strip() == "":
        st.warning("Please enter the model name.")
    else:
        age = 2025 - year
        price_per_km = asking_price / kms_driven if kms_driven > 0 else 0
        input_df = pd.DataFrame({
            'Brand': [brand],
            'Condition': [condition],
            'Fuel': [fuel],
            'KMs Driven': [kms_driven],
            'Model': [model_name],
            'Price': [asking_price],
            'Registered City': [registered_city],
            'Transaction Type': [transaction],
            'Year': [year],
            'Price Per KM': [price_per_km],
            'Age': [age]
        })

        try:
            prediction = model.predict(input_df)[0]
            st.markdown("### 🧠 Predicted Resale Price")
            st.success(f"💰 Estimated Price: **PKR {prediction:,.0f}**")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check your input values")
