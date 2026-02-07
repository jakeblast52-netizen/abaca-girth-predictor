import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import base64

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Abaca Girth Prediction System",
    layout="centered"
)

# ===============================
# BACKGROUND + OVERLAY
# ===============================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(150, 170, 145, 1), rgba(25, 45, 16, 0.65)),
                url(data:image/jpg;base64,{encoded});
            background-size: cover;
            background-attachment: fixed;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: #ffffff !important;
        }}

        .block-container {{
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")

# ===============================
# LOAD MODEL
# ===============================
with open("abaca_rf_model.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_cols = model_package["features"]

# ===============================
# HEADER
# ===============================
col1, col2 = st.columns([1, 4])

with col1:
    st.image("logo.png", width=110)

with col2:
    st.title("🌱 Abaca Girth Prediction System")
    st.caption("Machine Learning–Based Agricultural Prediction Tool")

st.markdown("---")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(
    ["📥 Input Parameters", "📊 Visualization", "ℹ️ About"]
)

# ===============================
# TAB 1: INPUT
# ===============================
with tab1:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.55);
        padding: 20px;
        border-radius: 15px;">
        """,
        unsafe_allow_html=True
    )

    st.subheader("Enter Abaca and Environmental Parameters")

    height_cm = st.slider("Plant Height (cm)", 50.0, 500.0, 200.0)
    leaf_count = st.slider("Leaf Count", 1, 20, 5)
    moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 60.0)
    soil_pH = st.slider("Soil pH", 3.0, 9.0, 6.5)
    temperature = st.slider("Temperature (°C)", 10.0, 45.0, 28.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    sun_shade = st.slider("Sun Shade (%)", 0.0, 100.0, 60.0)

    if st.button("🌿 Predict Abaca Girth"):
        input_data = pd.DataFrame([[ 
            height_cm,
            leaf_count,
            moisture,
            soil_pH,
            temperature,
            humidity,
            sun_shade
        ]], columns=feature_cols)

        predicted_log = model.predict(input_data)[0]
        predicted_girth = max(np.expm1(predicted_log), 0.5)

        st.metric(
            label="Predicted Abaca Girth (cm)",
            value=f"{predicted_girth:.2f}"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# TAB 2: VISUALIZATION
# ===============================
with tab2:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.55);
        padding: 20px;
        border-radius: 15px;">
        """,
        unsafe_allow_html=True
    )

    st.subheader("Feature Importance (Random Forest)")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.rcParams.update({
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white"
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        np.array(feature_cols)[sorted_idx],
        importances[sorted_idx],
        color="#7CFC98"
    )

    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Contribution to Girth Prediction")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# TAB 3: ABOUT
# ===============================
with tab3:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.55);
        padding: 20px;
        border-radius: 15px;">
        """,
        unsafe_allow_html=True
    )

    st.subheader("About the System")

    st.write(
        """
        This application predicts **Abaca (Musa textilis) plant girth**
        using a **Random Forest Regression** model trained on real field data.

        The system considers plant growth parameters and environmental
        conditions to support **agricultural decision-making and research**.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 | Abaca Girth Prediction System 🌾")
