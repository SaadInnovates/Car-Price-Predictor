import streamlit as st
import pandas as pd
from joblib import load
import os
import gdown
import sklearn
import sys
from importlib.metadata import version

# --- Version Compatibility Solution ---
def patch_sklearn():
    """Permanent solution for _RemainderColsList error"""
    try:
        # Try modern import first
        from sklearn.compose import ColumnTransformer
    except AttributeError:
        try:
            # Apply compatibility patch
            import sklearn.compose._column_transformer
            if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
                setattr(sklearn.compose._column_transformer, '_RemainderColsList', list)
                st.toast("Applied sklearn compatibility patch", icon="🔧")
        except Exception as e:
            st.error(f"CRITICAL VERSION ERROR: {str(e)}")
            st.error(f"Current scikit-learn: v{version('scikit-learn')}")
            st.error("Required version: 1.0.x or 1.1.x")
            st.stop()

# --- Robust Model Loader ---
@st.cache_resource(ttl=24*3600)
def load_model():
    MODEL_PATH = "car_price_model.joblib"
    GDRIVE_ID = "183946cS8Mjmcz7-qiVGU21_84iVDr5ZH"
    
    # Apply version fix before loading
    patch_sklearn()
    
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (20-30MB)..."):
            try:
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=True)
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                st.stop()
    
    # Load model with version safety
    try:
        return load(MODEL_PATH)
    except Exception as e:
        st.error("MODEL LOADING FAILED")
        st.error(f"Error: {str(e)}")
        st.error("Try one of these solutions:")
        st.code("""
        # Solution 1: Downgrade sklearn
        pip install scikit-learn==1.0.2
        
        # Solution 2: Use in Colab first
        !pip install scikit-learn==1.0.2
        !gdown --id 183946cS8Mjmcz7-qiVGU21_84iVDr5ZH
        """)
        st.stop()

# --- Initialize App ---
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
