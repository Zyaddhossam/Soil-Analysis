import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page title and layout
st.set_page_config(page_title="Soil Fertility Prediction", layout="centered")

# Title and description
st.title("Soil Fertility Prediction App")
st.write("""
Enter the soil parameters below to predict soil fertility using a trained Random Forest model.
The model expects 12 features: Nitrogen (N), Phosphorus (P), Potassium (K), pH, Electrical Conductivity (EC),
Organic Carbon (OC), Sulfur (S), Zinc (Zn), Iron (Fe), Copper (Cu), Manganese (Mn), and Boron (B).
""")

# Load the trained model
model = joblib.load('random_forest_model.joblib')
feature_names = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

valid_ranges = {
    'N': (0, 400), 'P': (0, 120), 'K': (0, 900), 'pH': (6, 9), 'EC': (0, 2),
    'OC': (0, 2), 'S': (0, 50), 'Zn': (0, 2), 'Fe': (0, 10), 'Cu': (0, 3),
    'Mn': (0, 20), 'B': (0, 3)
}

st.subheader("Input Soil Parameters")
with st.form(key="soil_form"):
    col1, col2 = st.columns(2)
    inputs = {}
    with col1:
        for feature in ['N', 'P', 'K', 'pH', 'EC', 'OC']:
            min_val, max_val = valid_ranges[feature]
            inputs[feature] = st.number_input(
                f"{feature} (Range: {min_val} - {max_val})",
                min_value=float(min_val),
                max_value=float(max_val),
                step=0.01,
                value=float(min_val),
                help=f"Enter value for {feature} within the range {min_val} to {max_val}."
            )
    with col2:
        for feature in ['S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']:
            min_val, max_val = valid_ranges[feature]
            inputs[feature] = st.number_input(
                f"{feature} (Range: {min_val} - {max_val})",
                min_value=float(min_val),
                max_value=float(max_val),
                step=0.01,
                value=float(min_val),
                help=f"Enter value for {feature} within the range {min_val} to {max_val}."
            )
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    try:
        input_data = [inputs[feature] for feature in feature_names]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        transformed_df = input_df.apply(lambda x: np.log10(x + 1e-10) if np.issubdtype(x.dtype, np.number) else x)
        
        prediction = model.predict(transformed_df)
        
        st.success(f"Prediction: {int(prediction[0])}")
        st.write("""
        **Interpretation**:
        - 0: Low fertility
        - 1: Medium fertility
        - 2: High fertility
        """)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

if st.button("Load Example Input"):
    example_input = [138, 8.6, 560, 7.46, 0.62, 0.7, 5.9, 0.24, 0.31, 0.77, 8.71, 0.11]
    for i, feature in enumerate(feature_names):
        st.session_state[f"soil_form_{feature}"] = example_input[i]

st.markdown("---")
st.write("Built with Streamlit. Model trained on soil fertility dataset.")