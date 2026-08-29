import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load saved model and scaler
model = joblib.load('attrition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Retrieve feature names expected by the scaler/model
expected_features = list(scaler.feature_names_in_)

st.title("👔 Employee Attrition Risk Predictor")
st.write("Enter employee details to evaluate attrition probability.")

# Input controls
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, value=5000)
    distance_from_home = st.slider("Distance From Home (km)", 1, 50, 5)

with col2:
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])

# Prediction logic
if st.button("Predict Attrition Risk"):
    # Create an empty input row populated with zeros matching all model features
    input_data = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)
    
    # Fill in matching numeric values if present in model features
    if 'Age' in input_data.columns:
        input_data['Age'] = age
    if 'MonthlyIncome' in input_data.columns:
        input_data['MonthlyIncome'] = monthly_income
    if 'DistanceFromHome' in input_data.columns:
        input_data['DistanceFromHome'] = distance_from_home
        
    # Set categorical encoded flags if present
    if overtime == "Yes" and 'OverTime_Yes' in input_data.columns:
        input_data['OverTime_Yes'] = 1
    elif overtime == "Yes" and 'OverTime' in input_data.columns:
        input_data['OverTime'] = 1
        
    dept_col = f"Department_{department}"
    if dept_col in input_data.columns:
        input_data[dept_col] = 1

    # Scale and predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Leaving** (Probability: {prob:.1%})")
    else:
        st.success(f"✅ **Low Risk of Leaving** (Probability: {prob:.1%})")