import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Crop AI", page_icon="🌱", layout="wide")

# ------------------ LOAD MODEL ------------------
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #dcedc1, #a8e6cf);
}

/* Title */
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2e7d32;
}

/* Card */
.card {
    background-color: #32cd32;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
}

/* Result */
.result {
    padding: 20px;
    border-radius: 15px;
    background: #e6ffe6;
    font-size: 22px;
    text-align: center;
    color: #2e7d32;
}

/* About box */
.about-box {
    background-color: #32cd32;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
}

/* Info box */
.info-box {
    background: #32cd32;
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #2e7d32;
    margin-bottom: 15px;
}

/* Button */
.stButton>button {
    background-color: #2e7d32;
    color: #32cd32;
    border-radius: 10px;
    padding: 10px 25px;
}

.stButton>button:hover {
    background-color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🌱 Crop AI")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Prediction", "ℹ️ About"])

# ------------------ HOME ------------------
if menu == "🏠 Home":
    st.markdown("<div class='title'>Smart Crop Recommendation System</div>", unsafe_allow_html=True)
    st.write("### 🌾 AI-powered crop suggestions based on soil & climate data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>🌱 Soil Analysis<br>Understand soil nutrients</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>🌦 Weather Impact<br>Climate-based decisions</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>🤖 AI Prediction<br>Accurate crop recommendations</div>", unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
elif menu == "📊 Prediction":

    st.markdown("## 🌾 Enter Soil & Weather Details")

    # 🔹 Info Section
    st.markdown("""
    <div class='info-box'>
    <b>🌱 Recommended Ranges (based on crop dataset)</b><br><br>

    Nitrogen (N): 0 – 140<br>
    Phosphorus (P): 5 – 145<br>
    Potassium (K): 5 – 205<br>
    Temperature: 8 – 45 °C<br>
    Humidity: 10 – 100 %<br>
    pH: 3.5 – 10<br>
    Rainfall: 20 – 300 mm
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)

    with col2:
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
        temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=45.0, value=25.0)

    with col3:
        humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
        ph = st.number_input("Soil pH", min_value=3.5, max_value=10.0, value=6.5)

    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0)

    # 🔹 Prediction
    if st.button("🚀 Predict Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        crop = le.inverse_transform(prediction)[0]

        st.markdown(f"<div class='result'>🌿 Recommended Crop: <b>{crop}</b></div>", unsafe_allow_html=True)

        st.info("💡 Tip: Maintain balanced nutrients for better yield.")

# ------------------ ABOUT ------------------
elif menu == "ℹ️ About":
    st.markdown("""
    <div class='about-box'>
        <h2>📘 About This Project</h2>
        <p>
        This is an AI-based Crop Recommendation System built using:
        <br><br>
        • Machine Learning (Random Forest)<br>
        • Streamlit for deployment<br>
        • Python for data processing<br><br>

        🎯 Goal: Help farmers choose the best crop based on soil & weather conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)