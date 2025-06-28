import streamlit as st
import subprocess
import sys
import os

# --- Step 1: Force sklearn 1.0.2 installation ---
try:
    import sklearn
    if sklearn.__version__ != "1.0.2":
        raise ImportError("Wrong sklearn version")
except:
    st.warning("Installing scikit-learn 1.0.2...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "scikit-learn==1.0.2",
        "--force-reinstall"
    ], check=True)
    st.rerun()

# --- Step 2: Apply patch BEFORE other imports ---
from patch_sklearn import *  # Must come before joblib import
from joblib import load

# --- Step 3: Model Loading ---
@st.cache_resource
def load_model():
    model_path = "car_price_model.joblib"
    if not os.path.exists(model_path):
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=183946cS8Mjmcz7-qiVGU21_84iVDr5ZH",
            model_path, quiet=True
        )
    return load(model_path)

# --- Rest of your Streamlit app remains unchanged ---
model = load_model()

st.set_page_config(page_title="Car Price Predictor", layout="centered", menu_items={
    'Get Help': 'https://github.com/your-repo',
    'Report a bug': "mailto:support@example.com"
})

# --- UI Components ---
st.title("🚗 Car Price Predictor")
st.caption("v2.1 | sklearn v" + version('scikit-learn'))

with st.expander("ℹ️ About this app"):
    st.write("""
    This app predicts used car prices using machine learning.
    For accurate results, please provide complete information.
    """)

# --- Data Input Section ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", [
            'Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'Mitsubishi', 'KIA', 'Other Brands',
            'Nissan', 'BMW', 'Mazda', 'Chevrolet', 'Daewoo', 'Hyundai', 'FAW',
            'Mercedes', 'Classic & Antiques', 'Lexus', 'Audi', 'Range Rover', 'Changan',
            'Porsche', 'Subaru', 'Land Rover', 'Others'
        ], index=0)

        condition = st.radio("Condition", ['Used', 'New'], horizontal=True)
        year = st.slider("Manufacturing Year", 1990, 2024, 2018, help="Select between 1990-2024")

    with col2:
        registered_city = st.selectbox("Registered City", sorted([
            'Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Multan', 'Other'
        ]))
        
        fuel = st.selectbox("Fuel Type", [
            'Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'
        ])
        
        kms_driven = st.number_input("KMs Driven", 
            min_value=0, 
            value=50000,
            step=1000,
            help="Total kilometers driven"
        )

    model_name = st.text_input("Model Name", placeholder="Corolla Altis, Civic Oriel etc.")
    submit = st.form_submit_button("Predict Price", type="primary")

# --- Prediction Logic ---
if submit:
    if not model_name.strip():
        st.warning("Please enter the car model name")
    else:
        try:
            # Feature engineering
            age = 2025 - year
            price_per_km = st.session_state.get('asking_price', 0) / kms_driven if kms_driven > 0 else 0
            
            # Create input dataframe
            input_df = pd.DataFrame([{
                'Brand': brand,
                'Condition': condition,
                'Fuel': fuel,
                'KMs Driven': kms_driven,
                'Model': model_name,
                'Registered City': registered_city,
                'Year': year,
                'Price Per KM': price_per_km,
                'Age': age
            }])

            # Prediction
            with st.spinner("🤖 Calculating fair price..."):
                prediction = model.predict(input_df)[0]
                
            # Display results
            st.balloons()
            st.success(f"### Estimated Value: PKR {prediction:,.0f}")
            
            with st.expander("Details"):
                st.json(input_df.iloc[0].to_dict())
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.code(str(e), language='python')
