import streamlit as st
import streamlit.components.v1 as components

# Optional: Set wide layout
st.set_page_config(layout="wide")

# Background image URLs from your GitHub repo
image_urls = [
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image1.jpg",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image2.png",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image3.jpg",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image4.jpg",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image5.jfif",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image5.png",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image6.png",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image7.jpg",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image8.jpg",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image9.jfif",
    "https://raw.githubusercontent.com/SaadInnovates/Car-Price-Predictor/main/images/image10.jfif"
]

components.html(f"""
<div class="bg" id="bg"></div>
<div class="overlay"></div>

<style>
.bg {{
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    background-size: cover;
    background-position: center;
    z-index: -1;
    transition: background-image 1s ease-in-out;
}}

.overlay {{
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: -1;
}}
</style>

<script>
const images = {js_array};
let index = 0;
const bg = document.getElementById("bg");

function rotateBackground() {{
    if (bg) {{
        bg.style.backgroundImage = `url('${{images[index]}}')`;
        index = (index + 1) % images.length;
    }}
}}

rotateBackground();
setInterval(rotateBackground, 5000);
</script>
""", height=0)


# Convert list to JS array
js_array = "[" + ", ".join([f'"{url}"' for url in image_urls]) + "]"




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
        transaction_type = st.selectbox("Transaction Type", ["Cash", "Installment/Lease"])
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
