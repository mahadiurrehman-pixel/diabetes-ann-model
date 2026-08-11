import streamlit as st
import numpy as np
import joblib
import os
import json
from tensorflow.keras.models import model_from_json

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #0f3460;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e94560 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #a8a8b3 !important;
    }
    h1 {
        background: linear-gradient(90deg, #e94560, #0f3460, #53a8b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 900 !important;
        text-align: center;
        padding: 10px 0;
    }
    h2, h3 {
        color: #53a8b6 !important;
        font-weight: 700 !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 25px;
        margin: 10px 0;
        backdrop-filter: blur(12px);
    }
    .stat-card {
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.15), rgba(83, 168, 182, 0.15));
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #e94560, #53a8b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #a8a8b3;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 5px;
    }
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(83, 168, 182, 0.4) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 10px 15px !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 15px rgba(233, 69, 96, 0.3) !important;
    }
    .stNumberInput label {
        color: #53a8b6 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #c23152 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 18px 40px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.6) !important;
    }
    .result-positive {
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.2), rgba(194, 49, 82, 0.1));
        border: 2px solid #e94560;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        animation: pulse-red 2s infinite;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.2), rgba(39, 174, 96, 0.1));
        border: 2px solid #2ed573;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(233, 69, 96, 0.3); }
        50% { box-shadow: 0 0 40px rgba(233, 69, 96, 0.6); }
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(46, 213, 115, 0.3); }
        50% { box-shadow: 0 0 40px rgba(46, 213, 115, 0.6); }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2ed573, #ffa502, #e94560) !important;
        border-radius: 10px !important;
    }
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #e94560 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #53a8b6 !important;
        font-weight: 600 !important;
    }
    .rec-card {
        background: rgba(83, 168, 182, 0.1);
        border-left: 4px solid #53a8b6;
        border-radius: 0 12px 12px 0;
        padding: 15px 20px;
        margin: 8px 0;
        color: #e0e0e0;
        font-size: 1rem;
    }
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #e94560, #53a8b6, transparent);
        border: none;
        margin: 30px 0;
        border-radius: 2px;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 20px;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 40px;
    }
    .footer a { color: #e94560; text-decoration: none; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .hero-section { text-align: center; padding: 20px 0 30px 0; }
    .hero-emoji {
        font-size: 4rem;
        animation: float 3s ease-in-out infinite;
        display: block;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    .hero-subtitle {
        color: #a8a8b3;
        font-size: 1.15rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL & SCALER — FIXED
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load architecture
    with open('model_architecture.json', 'r') as f:
        model_json = json.load(f)

    # Rebuild model
    model = model_from_json(model_json)

    # Load weights
    model.load_weights('model_weights.weights.h5')

    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Load scaler
    scaler = joblib.load('scaler.pkl')

    return model, scaler

# ==========================================
# GLOBAL VARIABLES — FIXED SCOPE
# ==========================================
model, scaler = load_artifacts()  # ← Yahan globally assign ho rahe hain


# ==========================================
# HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-section">
    <span class="hero-emoji">🩺</span>
</div>
""", unsafe_allow_html=True)

st.title("Diabetes Prediction AI")

st.markdown("""
<div class="hero-section">
    <div class="hero-subtitle">Powered by Artificial Neural Network</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 About the Model")

    st.markdown("""
    <div class="glass-card">
        <p style="color: #a8a8b3; font-size: 0.95rem;">
        This AI system uses a <b style="color: #e94560;">Deep Learning ANN</b>
        trained on the Pima Indians Diabetes Dataset to predict
        diabetes risk with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance")

    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">81%</div>
        <div class="stat-label">Recall (Sensitivity)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">68%</div>
        <div class="stat-label">F1 Score</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">82%</div>
        <div class="stat-label">ROC AUC Score</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🔬 Tech Stack")
    st.markdown("""
    <div class="glass-card">
        <p style="color: #e0e0e0;">
        ⚡ TensorFlow / Keras<br>
        📊 Scikit-Learn<br>
        ⚖️ SMOTE + Class Weights<br>
        🌐 Streamlit<br>
        🐍 Python 3.12
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="color: #666; font-size: 0.8rem;">
        Built with ❤️ by<br>
        <a href="https://github.com/mahadiurrehman-pixel"
           style="color: #e94560; text-decoration: none; font-weight: bold;">
        Mahadiur Rehman
        </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# INPUT SECTION
# ==========================================
st.markdown("""
<div class="main-card">
    <h3 style="text-align: center; margin-bottom: 5px;">📝 Patient Information</h3>
    <p style="text-align: center; color: #a8a8b3; font-size: 0.9rem;">
    Enter the clinical measurements below for prediction
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #e94560; text-align: center;">🫀 Vitals</h4>
    </div>
    """, unsafe_allow_html=True)

    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0, max_value=20, value=1,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "🍬 Glucose (mg/dL)",
        min_value=0, max_value=300, value=120,
        help="Plasma glucose concentration"
    )
    blood_pressure = st.number_input(
        "💉 Blood Pressure (mm Hg)",
        min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure"
    )
    skin_thickness = st.number_input(
        "📏 Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #53a8b6; text-align: center;">📋 Measurements</h4>
    </div>
    """, unsafe_allow_html=True)

    insulin = st.number_input(
        "💊 Insulin (mu U/ml)",
        min_value=0, max_value=900, value=80,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index"
    )
    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5, step=0.01,
        help="Genetic diabetes risk score"
    )
    age = st.number_input(
        "🎂 Age (years)",
        min_value=1, max_value=120, value=30,
        help="Patient age in years"
    )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ==========================================
# PREDICT BUTTON
# ==========================================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button(
        "🔍 ANALYZE & PREDICT",
        use_container_width=True
    )

# ==========================================
# PREDICTION LOGIC — FIXED SCOPE
# ==========================================
if predict_clicked:

    # Input array
    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]])

    # Scale — ab scaler globally available hai
    input_scaled = scaler.transform(input_data)

    # Predict — ab model globally available hai
    prediction_prob = float(model.predict(input_scaled, verbose=0)[0][0])
    prediction = 1 if prediction_prob > 0.5 else 0

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Results")
    st.markdown("")

    # Results Cards
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{prediction_prob*100:.1f}%</div>
            <div class="stat-label">Diabetes Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        if prediction_prob < 0.3:
            risk_emoji = "🟢"
            risk_text = "LOW RISK"
            risk_color = "#2ed573"
        elif prediction_prob < 0.7:
            risk_emoji = "🟡"
            risk_text = "MEDIUM RISK"
            risk_color = "#ffa502"
        else:
            risk_emoji = "🔴"
            risk_text = "HIGH RISK"
            risk_color = "#e94560"

        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2.5rem;">{risk_emoji}</div>
            <div style="color: {risk_color}; font-size: 1.3rem;
                        font-weight: 900; letter-spacing: 2px;">
                {risk_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        confidence = max(prediction_prob, 1 - prediction_prob) * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{confidence:.1f}%</div>
            <div class="stat-label">Model Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Risk Meter
    st.markdown("""
    <div class="risk-meter" style="background: rgba(255,255,255,0.05);
         border-radius: 16px; padding: 20px;
         border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: #53a8b6; text-align: center;">📈 Risk Meter</h4>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(prediction_prob))

    st.markdown("")

    # Result Message
    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <h2 style="color: #e94560; margin: 0;">⚠️ DIABETES RISK DETECTED</h2>
            <p style="color: #e0e0e0; font-size: 1.1rem; margin-top: 15px;">
            Based on the clinical measurements, our AI model indicates a
            <b style="color: #e94560;">{prediction_prob*100:.1f}%</b> probability
            of diabetes. Please consult a healthcare professional immediately.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <h2 style="color: #2ed573; margin: 0;">✅ LOW DIABETES RISK</h2>
            <p style="color: #e0e0e0; font-size: 1.1rem; margin-top: 15px;">
            Great news! Our AI model indicates a
            <b style="color: #2ed573;">{(1-prediction_prob)*100:.1f}%</b> probability
            of being diabetes-free. Keep maintaining a healthy lifestyle!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Health Recommendations
    st.markdown("## 💡 Personalized Health Recommendations")
    st.markdown("")

    recommendations = []

    if glucose > 140:
        recommendations.append({
            "icon": "🍎", "title": "High Glucose Alert",
            "text": "Your glucose level is elevated. Monitor blood sugar regularly."
        })
    if bmi > 30:
        recommendations.append({
            "icon": "🏃", "title": "BMI Above Normal",
            "text": "Consider a weight management program with regular exercise."
        })
    if blood_pressure > 130:
        recommendations.append({
            "icon": "💉", "title": "Elevated Blood Pressure",
            "text": "Monitor blood pressure regularly. Reduce sodium intake."
        })
    if age > 45:
        recommendations.append({
            "icon": "👨‍⚕️", "title": "Age Factor",
            "text": "Diabetes risk increases with age. Schedule annual checkups."
        })
    if dpf > 0.8:
        recommendations.append({
            "icon": "🧬", "title": "Genetic Risk Factor",
            "text": "Your family history indicates higher risk. Be proactive."
        })
    if insulin > 200:
        recommendations.append({
            "icon": "💊", "title": "High Insulin Level",
            "text": "Elevated insulin may indicate resistance. Consult a doctor."
        })

    if recommendations:
        for rec in recommendations:
            st.markdown(f"""
            <div class="rec-card">
                <span style="font-size: 1.3rem;">{rec['icon']}</span>
                <b style="color: #53a8b6;"> {rec['title']}</b><br>
                <span style="color: #a8a8b3;">{rec['text']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rec-card" style="border-left-color: #2ed573;">
            <span style="font-size: 1.3rem;">✅</span>
            <b style="color: #2ed573;"> All Clear!</b><br>
            <span style="color: #a8a8b3;">
            Your values look great! Keep maintaining your healthy lifestyle.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Input Summary
    st.markdown("## 📋 Input Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown(f"""
        <div class="glass-card">
            <table style="width: 100%; color: #e0e0e0;">
                <tr><td style="padding:8px; color:#53a8b6;">🤰 Pregnancies</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{pregnancies}</td></tr>
                <tr><td style="padding:8px; color:#53a8b6;">🍬 Glucose</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{glucose} mg/dL</td></tr>
                <tr><td style="padding:8px; color:#53a8b6;">💉 Blood Pressure</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{blood_pressure} mm Hg</td></tr>
                <tr><td style="padding:8px; color:#53a8b6;">📏 Skin Thickness</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{skin_thickness} mm</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with summary_col2:
        st.markdown(f"""
        <div class="glass-card">
            <table style="width: 100%; color: #e0e0e0;">
                <tr><td style="padding:8px; color:#53a8b6;">💊 Insulin</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{insulin} mu U/ml</td></tr>
                <tr><td style="padding:8px; color:#53a8b6;">⚖️ BMI</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{bmi} kg/m²</td></tr>
                <tr><td style="padding:8px; color:#53a8b6;">🧬 DPF</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{dpf}</td></tr>
                <tr><td style="padding:8px; color:#53a8b6;">🎂 Age</td>
                    <td style="padding:8px; text-align:right; font-weight:bold;">{age} years</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <p>
    ⚠️ <b>Medical Disclaimer:</b> This is an AI-powered predictive tool for
    educational purposes only. It is <b>NOT</b> a substitute for professional
    medical diagnosis. Always consult a qualified healthcare professional.
    </p>
    <p style="margin-top: 15px;">
    🩺 Diabetes Prediction AI ·
    <a href="https://github.com/mahadiurrehman-pixel/diabetes-ann-model">GitHub</a> ·
    © 2024 Mahadiur Rehman
    </p>
</div>
""", unsafe_allow_html=True)
