# Car Price Predictor
A machine learning-powered web app to estimate the fair market price of used cars in Pakistan. Built using Streamlit, this project enables users to input car details and receive an accurate price prediction.
# Features
- Predicts car price using a trained CatBoost regression model
  
- Clean and responsive Streamlit web UI

- Inputs include brand, model, year, fuel type, city, condition, kilometers driven, etc.

- Model is hosted and downloaded securely from Google Drive

### 📊 Tech Stack

| Tool                 | Purpose                          |
|----------------------|----------------------------------|
| Python               | Core programming language        |
| Pandas               | Data manipulation                |
| Streamlit            | Web application interface        |
| RandomForestRegressor| Regression model for prediction  |
| Joblib               | Model serialization              |
| gdown                | Secure model file download       |

# 🧠 Machine Learning Model
Trained using historical car listings data
Preprocessing includes feature engineering for:
Age (2025 - Year)
Price per km

Model: RandomForestRegressor (robust on categorical data and performs well with minimal tuning)
R2 score : 0.92 
Cross Valid Score : 0.90
