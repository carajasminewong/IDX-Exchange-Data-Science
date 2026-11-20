import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. Load the trained model (cached so it only loads once)
@st.cache_resource
def load_model():
    return joblib.load("gb_dt_model (1).joblib")  # Replace with your exact model filename

model = load_model()

# App Title & Description
st.title("House Price Predictor")
st.write("Enter the property details below to estimate the market price.")

# 2. User Inputs
col1, col2 = st.columns(2)

with col1:
    living_area = st.number_input("Living Area (sq ft)", min_value=100, max_value=20000, value=1800, step=50)
    beds = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)

with col2:
    baths = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    lot_size = st.number_input("Lot Size (sq ft)", min_value=0, max_value=500000, value=5000, step=100)

# 3. Predict and Output
if st.button("Predict Price", type="primary"):
    # Default assumptions for features not collected in the UI
    default_year_built = 2000
    current_year = datetime.now().year
    median_living_area = 1800.0  # Replace with your training set's median LivingArea if known

    # Calculate engineered ratios from user inputs
    lot_area_ratio = living_area / lot_size if lot_size > 0 else 0.0
    bed_bath_ratio = beds / baths if baths > 0 else float(beds)
    relative_size = living_area / median_living_area

    # Assemble DataFrame matching the exact schema and transformations used during training
    input_data = pd.DataFrame([{
        # Log-transformed skewed numerical features
        "LivingArea": np.log1p(living_area),
        "LotSizeSquareFeet": np.log1p(lot_size),
        "LotAreaRatio": np.log1p(lot_area_ratio),
        "BedBathRatio": np.log1p(bed_bath_ratio),
        "RelativeSize": np.log1p(relative_size),

        # Untransformed numerical features
        "BedroomsTotal": beds,
        "BathroomsTotalInteger": baths,
        "YearBuilt": default_year_built,
        "AgeOfProperty": current_year - default_year_built,
        "Stories": 1,

        # Categorical features (handled by TargetEncoder / SimpleImputer)
        "City": "Unknown",
        "CountyOrParish": "Unknown",
        "DistrictName": "Unknown",
    }])

    # Predict log-price and convert back to actual dollars
    log_prediction = model.predict(input_data)[0]
    dollar_prediction = np.expm1(log_prediction)

    st.success(f"Estimated Price: **${dollar_prediction:,.2f}**")