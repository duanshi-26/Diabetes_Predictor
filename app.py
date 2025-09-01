import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'diabetes_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Set a custom page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Header Section
st.title("🩺 Diabetes Prediction App")
st.markdown("""
Welcome to the **Diabetes Prediction App**!  
Provide medical details in the sidebar to predict whether a person is likely to have diabetes.
""")
st.divider()

# Sidebar for Input
st.sidebar.header("Enter Patient Details")
st.sidebar.markdown("Fill in the required medical data below:")

# Input fields with better organization
with st.sidebar.form("input_form"):
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)
    submitted = st.form_submit_button("Predict")

# Preprocessing function to match training data format
def preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    # Create a dataframe for easier manipulation
    input_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age],
    })

    # Add one-hot encoded categorical variables with default values
    input_df["NewBMI_Obesity 1"] = 0
    input_df["NewBMI_Obesity 2"] = 0
    input_df["NewBMI_Obesity 3"] = 0
    input_df["NewBMI_Overweight"] = 0
    input_df["NewBMI_Underweight"] = 0
    input_df["NewInsulinScore_Normal"] = 0
    input_df["NewGlucose_Low"] = 0
    input_df["NewGlucose_Normal"] = 0
    input_df["NewGlucose_Overweight"] = 0
    input_df["NewGlucose_Secret"] = 0

    # Ensure all features are in the correct order
    feature_order = model.feature_names_in_
    input_df = input_df[feature_order]

    return input_df

# Main Section
if submitted:
    # Preprocess input data
    input_data = preprocess_input(
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age
    )

    # Predict and display results
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display prediction result
    if prediction[0] == 1:
        st.error("🚨 The model predicts that the person **has diabetes**.")
    else:
        st.success("✅ The model predicts that the person **does not have diabetes**.")

    # Display prediction probabilities
    st.subheader("Prediction Probability:")
    prob_df = pd.DataFrame(prediction_proba, columns=["Non-Diabetic", "Diabetic"])
    st.bar_chart(prob_df.T)
