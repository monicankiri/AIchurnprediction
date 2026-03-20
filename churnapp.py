import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
with open("model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Predict if a customer is likely to leave the service.")

# Inputs
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("Monthly Charges", 0.0, 1000.0)

# Prepare input
input_data = pd.DataFrame({
    'SeniorCitizen': [senior],
    'tenure': [tenure],
    'Contract': [contract],
    'MonthlyCharges': [monthly]
})

# Predict
if st.button("Predict Churn"):
    prediction = model_pipeline.predict(input_data)[0]
    probability = model_pipeline.predict_proba(input_data)[0][1]  # probability of churn
    
    st.write(f"Probability of churn: {probability*100:.2f}%")
    
    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")