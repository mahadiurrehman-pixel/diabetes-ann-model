import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = load_model('diabetes_ann_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# Title
st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict diabetes using Artificial Neural Network")
st.markdown("---")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app uses an **ANN model** trained on the "
    "Pima Indians Diabetes Dataset to predict "
    "diabetes probability."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write("**Model:** ANN with Class Weight")
st.sidebar.write("**Recall:** 81%")
st.sidebar.write("**F1 Score:** 68%")

# Input form
st.markdown("### 📝 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", 
        min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)", 
        min_value=0, max_value=300, value=120
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)", 
        min_value=0, max_value=200, value=70
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", 
        min_value=0, max_value=100, value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (mu U/ml)", 
        min_value=0, max_value=900, value=80
    )
    bmi = st.number_input(
        "BMI (kg/m²)", 
        min_value=0.0, max_value=70.0, value=25.0, step=0.1
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", 
        min_value=0.0, max_value=3.0, value=0.5, step=0.01
    )
    age = st.number_input(
        "Age (years)", 
        min_value=1, max_value=120, value=30
    )

st.markdown("---")

# Predict button
if st.button("🔍 Predict Diabetes", use_container_width=True):
    
    # Input array
    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0
    
    # Display results
    st.markdown("### 📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Diabetes Probability", 
            f"{prediction_prob*100:.2f}%"
        )
    
    with col2:
        risk_level = "🟢 Low Risk" if prediction_prob < 0.3 else \
                     "🟡 Medium Risk" if prediction_prob < 0.7 else \
                     "🔴 High Risk"
        st.metric("Risk Level", risk_level)
    
    # Result message
    if prediction == 1:
        st.error(
            "⚠️ **HIGH RISK OF DIABETES DETECTED!**\n\n"
            "Please consult a doctor immediately for "
            "proper diagnosis and treatment."
        )
    else:
        st.success(
            "✅ **LOW RISK OF DIABETES**\n\n"
            "Keep maintaining a healthy lifestyle! "
            "Regular check-ups are recommended."
        )
    
    # Progress bar
    st.markdown("### 📈 Risk Meter")
    st.progress(float(prediction_prob))
    
    # Recommendations
    st.markdown("### 💡 Health Recommendations")
    
    recommendations = []
    
    if glucose > 140:
        recommendations.append("🍎 Monitor blood sugar levels regularly")
    if bmi > 30:
        recommendations.append("🏃 Consider weight management program")
    if blood_pressure > 130:
        recommendations.append("💊 Check blood pressure regularly")
    if age > 45:
        recommendations.append("👨‍⚕️ Regular medical checkups recommended")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.write("✅ Your values look good! Keep it up!")

st.markdown("---")
st.caption("⚠️ This is a predictive model, not a medical diagnosis. Always consult a doctor.")