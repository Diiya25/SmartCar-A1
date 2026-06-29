# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartCar AI", page_icon="🚗", layout="wide")

# ── IMAGE LOADER ───────────────────────────────────────────────────────────────
HERO_IMG = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/2021_Toyota_Fortuner_Legender_4x2_AT_%28front%29.jpg/640px-2021_Toyota_Fortuner_Legender_4x2_AT_%28front%29.jpg"
DEFAULT_CAR_IMG = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Toyota_Corolla_E210_India_2019_front.jpg/320px-Toyota_Corolla_E210_India_2019_front.jpg"

def get_car_image(car_name: str):
    filename = car_name.lower().replace(" ", "_") + ".jpg"
    path = f"images/{filename}"
    if os.path.exists(path):
        return path
    return DEFAULT_CAR_IMG

# ── THEME ──────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@400;500;600;700;800&family=Barlow+Condensed:wght@700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif !important;
}

.stApp {
    background: #FAFAFA;
    color: #111111;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding: 0 2.5rem 3rem;
    max-width: 1280px;
}

label, .stSlider label {
    color: #888888 !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stTextInput div div input,
.stNumberInput div div input,
.stSelectbox div div {
    background: #FFFFFF !important;
    color: #111111 !important;
    border: 1.5px solid #EBEBEB !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

.stSlider div div div div {
    background: #C0392B !important;
}

.hero {
    position: relative;
    width: 100%;
    min-height: 300px;
    background: radial-gradient(ellipse at 65% 50%, rgba(192,57,43,0.10) 0%, transparent 65%),
                linear-gradient(160deg, #fff 0%, #FDF1F0 55%, #fff 100%);
    border-bottom: 2px solid #EBEBEB;
    display: flex;
    align-items: center;
    padding: 2.5rem 2rem 2rem;
    margin: 0 -2.5rem 2rem -2.5rem;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: repeating-linear-gradient(
        -45deg,
        transparent,
        transparent 40px,
        rgba(192,57,43,0.025) 40px,
        rgba(192,57,43,0.025) 41px
    );
    pointer-events: none;
}

.hero-left {
    flex: 1;
    z-index: 2;
    padding-right: 2rem;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FDECEA;
    border: 1.5px solid rgba(192,57,43,0.3);
    color: #C0392B;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 5px 16px;
    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 86px;
    font-weight: 400;
    color: #111111;
    letter-spacing: 2px;
    line-height: 0.88;
    text-transform: uppercase;
    margin-bottom: 0;
}

.hero-title span {
    color: #C0392B;
}

.hero-subtitle {
    color: #888888;
    font-size: 14px;
    font-weight: 500;
    margin-top: 16px;
    letter-spacing: 0.4px;
}

.hero-chips {
    margin-top: 22px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.hero-chip {
    background: #fff;
    border: 1.5px solid #EBEBEB;
    color: #555;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 5px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.hero-car-wrap {
    flex: 0 0 460px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 2;
}

.hero-car-shadow {
    position: absolute;
    width: 340px;
    height: 40px;
    background: radial-gradient(ellipse, rgba(192,57,43,0.22) 0%, transparent 70%);
    bottom: 0px;
    left: 50%;
    transform: translateX(-50%);
    filter: blur(14px);
    animation: shadowPulse 4s ease-in-out infinite;
}

@keyframes shadowPulse {
    0%,100% { opacity: 0.7; transform: translateX(-50%) scaleX(1); }
    50% { opacity: 1; transform: translateX(-50%) scaleX(1.12); }
}

.hero-car-img {
    width: 430px;
    max-width: 100%;
    filter: drop-shadow(0 16px 40px rgba(192,57,43,0.18)) drop-shadow(0 2px 8px rgba(0,0,0,0.10));
    animation: floatCar 4.5s ease-in-out infinite;
    transform-origin: center bottom;
    border-radius: 12px;
}

@keyframes floatCar {
    0%,100% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
}

.divider {
    height: 2px;
    background: linear-gradient(90deg, #C0392B 0%, rgba(192,57,43,0.2) 60%, transparent 100%);
    border: none;
    margin: 1.5rem 0;
    border-radius: 2px;
}

.section-label {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 16px;
    letter-spacing: 3px;
    color: #888;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-label::before {
    content: "";
    display: inline-block;
    width: 20px;
    height: 3px;
    background: #C0392B;
    border-radius: 2px;
    flex-shrink: 0;
}

.metric-card {
    background: #fff;
    border-radius: 16px;
    padding: 20px;
    border: 1.5px solid #EBEBEB;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}

.metric-card.hi {
    border-color: rgba(192,57,43,0.35);
    background: linear-gradient(135deg, #fff 0%, #FDECEA 100%);
    box-shadow: 0 4px 20px rgba(192,57,43,0.12), 0 0 0 1px rgba(192,57,43,0.1);
}

.mc-model {
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #aaa;
}

.mc-label {
    font-size: 11px;
    color: #999;
    margin-top: 3px;
}

.mc-val {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 36px;
    color: #C0392B;
    margin-top: 6px;
    line-height: 1;
    letter-spacing: 1px;
}

.result-card {
    background: #fff;
    border-radius: 16px;
    padding: 22px;
    border: 1.5px solid #EBEBEB;
    margin-bottom: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}

.result-card.best {
    border-color: rgba(192,57,43,0.4);
    background: linear-gradient(135deg, #fff 0%, #FDECEA 100%);
    box-shadow: 0 6px 28px rgba(192,57,43,0.14), 0 0 0 1px rgba(192,57,43,0.1);
}

.rc-model {
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #aaa;
    margin-bottom: 4px;
}

.rc-price {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 44px;
    color: #C0392B;
    line-height: 1;
    letter-spacing: 1px;
}

.rc-pill {
    display: inline-block;
    background: #C0392B;
    color: #fff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 4px 14px;
    margin-top: 10px;
}

.empty-state {
    background: #fff;
    border-radius: 16px;
    border: 1.5px dashed #EBEBEB;
    padding: 52px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

.empty-icon {
    font-size: 40px;
    margin-bottom: 12px;
}

.empty-text {
    font-size: 14px;
    color: #888;
    font-weight: 500;
}

.empty-cta {
    color: #C0392B;
    font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #F3F4F6;
    border-radius: 12px;
    padding: 5px;
    border: 1.5px solid #EBEBEB;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #888;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 28px;
    letter-spacing: 0.3px;
}

.stTabs [aria-selected="true"] {
    background: #C0392B !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(192,57,43,0.35) !important;
}

div[data-testid="stFormSubmitButton"] {
    width: 100%;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%) !important;
    color: #fff !important;
    width: 100% !important;
    padding: 15px 24px !important;
    border-radius: 12px !important;
    border: none !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-weight: 400 !important;
    font-size: 20px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(192,57,43,0.35), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    transition: all 0.2s !important;
    display: block !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 8px 32px rgba(192,57,43,0.5) !important;
    transform: translateY(-1px) !important;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}

.car-card {
    background: #fff;
    border-radius: 20px;
    border: 1.5px solid #EBEBEB;
    overflow: hidden;
    animation: fadeSlideUp 0.45s ease forwards;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.car-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow:
        0 25px 70px rgba(99,102,241,0.30),
        0 0 40px rgba(99,102,241,0.20);
}

.car-img-container {
    overflow: hidden;
    height: 180px;
    background: linear-gradient(135deg, #F9FAFB 0%, #F1F3F5 100%);
    display: flex;
    align-items: center;
    justify-content: center;
}

.car-img-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s ease;
}

.car-img-container:hover img {
    transform: scale(1.06);
}

.car-body {
    padding: 16px 18px 20px;
}

.car-brand {
    font-size: 10px;
    font-weight: 800;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 2px;
}

.car-name {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 26px;
    color: #111111;
    line-height: 1.05;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.car-price {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 22px;
    color: #C0392B;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.car-rating {
    font-size: 13px;
    font-weight: 700;
    color: #F59E0B;
    margin-bottom: 6px;
}

.car-desc {
    font-size: 12px;
    color: #666;
    line-height: 1.55;
    margin-bottom: 12px;
}

.badge {
    display: inline-block;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    color: #666;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}

.badge-petrol {
    display: inline-block;
    background: #FDECEA;
    border: 1px solid rgba(192,57,43,0.3);
    color: #C0392B;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}

.badge-diesel {
    display: inline-block;
    background: #EFF6FF;
    border: 1px solid rgba(59,130,246,0.3);
    color: #2563EB;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}

.badge-electric {
    display: inline-block;
    background: #ECFDF5;
    border: 1px solid rgba(16,185,129,0.3);
    color: #059669;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}

.no-result {
    text-align: center;
    padding: 48px 24px;
    border: 1.5px dashed #EBEBEB;
    border-radius: 16px;
    color: #aaa;
    font-size: 14px;
    font-weight: 500;
    background: #fff;
}

.stAlert {
    border-radius: 12px !important;
    font-weight: 500 !important;
}

.footer {
    text-align: center;
    color: #bbb;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 20px 0 4px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-tag">AI &nbsp;&nbsp; ML Powered</div>
    <div class="hero-title">SMART<span>CAR</span><br>AI</div>
    <div class="hero-subtitle">AI-Powered Vehicle Intelligence • Predict • Analyze • Recommend</div>
    <div class="hero-chips">
      <span class="hero-chip">CarDekho Dataset</span>
      <span class="hero-chip">Linear Regression</span>
      <span class="hero-chip">Random Forest</span>
      <span class="hero-chip">Rule-Based Recommender</span>
    </div>
  </div>
  <div class="hero-car-wrap">
    <div class="hero-car-shadow"></div>
    <img class="hero-car-img" src="{HERO_IMG}" alt="SmartCar Hero"/>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# ── DATA / MODEL ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("car_data.csv")
    car_keywords = "ritz,sx4,ciaz,wagon,swift,brezza,ertiga,dzire,alto,baleno,ignis,omni,fortuner,innova,corolla,etios,camry,land cruiser,i20,i10,grand,eon,xcent,creta,verna,elantra,city,brio,amaze,jazz"
    df = df[df["Car_Name"].str.lower().str.contains("|".join(car_keywords), na=False)]
    df = df.drop(["Car_Name", "Owner"], axis=1)
    df.replace({"Fuel_Type": {"Petrol": 0, "Diesel": 1, "CNG": 2}}, inplace=True)
    df.replace({"Seller_Type": {"Dealer": 0, "Individual": 1}}, inplace=True)
    df.replace({"Transmission": {"Manual": 0, "Automatic": 1}}, inplace=True)

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=2)
    rf.fit(X_train, y_train)

    lr_r2 = r2_score(y_test, lr.predict(X_test))
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

    return lr, rf, lr_r2, lr_mae, rf_r2, rf_mae

@st.cache_data
def load_car_data():
    return pd.read_csv("car_recommendations.csv")

tab1, tab2 = st.tabs(["Price Predictor", "Car Recommender"])

# ── PRICE PREDICTOR ───────────────────────────────────────────────────────────
with tab1:
    lr_model, rf_model, lr_r2, lr_mae, rf_r2, rf_mae = load_and_train()

    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    def metric_card(col, model, label, value, hi=False):
        cls = "metric-card hi" if hi else "metric-card"
        col.markdown(
            f"""
            <div class="{cls}">
              <div class="mc-model">{model}</div>
              <div class="mc-label">{label}</div>
              <div class="mc-val">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    metric_card(s1, "Linear Regression", "R Score", f"{lr_r2:.3f}")
    metric_card(s2, "Linear Regression", "Avg Error", f"₹{lr_mae:.2f}L")
    metric_card(s3, "Random Forest", "R Score", f"{rf_r2:.3f}", hi=True)
    metric_card(s4, "Random Forest", "Avg Error", f"₹{rf_mae:.2f}L", hi=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, _, right = st.columns([1.3, 0.05, 1])

    with left:
        st.markdown('<div class="section-label">Car Details</div>', unsafe_allow_html=True)
        with st.form("price_form"):
            c1, c2 = st.columns(2)
            with c1:
                year = st.selectbox("Year", list(range(2003, 2026)), index=12)
                present_price = st.number_input("Showroom Price (Lakh)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
                kms_driven = st.number_input("Kilometres Driven", min_value=500, max_value=500000, value=30000, step=1000)
            with c2:
                fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
                seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Get Prediction")

    with right:
        st.markdown('<div class="section-label">Predicted Price</div>', unsafe_allow_html=True)
        if submitted:
            fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            trans_map = {"Manual": 0, "Automatic": 1}
            inp = np.array([[year, present_price, kms_driven, fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]]])
            lr_pred = max(0, lr_model.predict(inp)[0])
            rf_pred = max(0, rf_model.predict(inp)[0])

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="rc-model">Linear Regression</div>
                  <div class="rc-price">{lr_pred:.2f} <span style="font-size:17px;font-weight:600;font-family:Barlow,sans-serif;">Lakh</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="result-card best">
                  <div class="rc-model">Random Forest</div>
                  <div class="rc-price">{rf_pred:.2f} <span style="font-size:17px;font-weight:600;font-family:Barlow,sans-serif;">Lakh</span></div>
                  <div class="rc-pill">Most Accurate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by ₹{diff:.2f}L. Best R²: {max(lr_r2, rf_r2):.3f}")
        else:
            st.markdown(
                """
                <div class="empty-state">
                  <div class="empty-icon">✨</div>
                  <div class="empty-text">Fill in car details and hit<br><span class="empty-cta">Get Prediction</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ── RECOMMENDER ────────────────────────────────────────────────────────────────
with tab2:
    df_cars = load_car_data()

    st.markdown('<div class="section-label">Find Your Perfect Car</div>', unsafe_allow_html=True)
    with st.form("reco_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            budget = st.selectbox(
                "Budget",
                ["under5L", "5-10L", "10-20L", "20L"],
                format_func=lambda x: {"under5L": "Under 5 Lakh", "5-10L": "5-10 Lakh", "10-20L": "10-20 Lakh", "20L": "Above 20 Lakh"}[x]
            )
            fuel = st.selectbox(
                "Fuel Type",
                ["any", "petrol", "diesel", "electric"],
                format_func=lambda x: {"any": "No Preference", "petrol": "Petrol", "diesel": "Diesel", "electric": "Electric"}[x]
            )
        with c2:
            usage = st.selectbox(
                "Usage",
                ["city", "highway", "both"],
                format_func=lambda x: {"city": "Mostly City", "highway": "Mostly Highway", "both": "Both"}[x]
            )
            family_size = st.selectbox(
                "Family Size",
                ["1-2", "3-4", "5"],
                format_func=lambda x: {"1-2": "1-2 People", "3-4": "3-4 People", "5": "5+ People"}[x]
            )
        with c3:
            priority = st.selectbox(
                "Priority",
                ["mileage", "comfort", "style", "performance"],
                format_func=lambda x: {"mileage": "Best Mileage", "comfort": "Comfort Space", "style": "Stylish Looks", "performance": "Performance"}[x]
            )
        st.markdown("<br>", unsafe_allow_html=True)
        search = st.form_submit_button("Find My Car")

    if search:
        filtered = df_cars.copy()

        if budget != "under5L":
            filtered = filtered[filtered["budget"] == budget]
        else:
            filtered = filtered[filtered["budget"] == budget]

        if fuel != "any":
            filtered = filtered[filtered["fuel"] == fuel]
        if usage:
            filtered = filtered[filtered["usage"] == usage]
        if family_size:
            filtered = filtered[filtered["family_size"] == family_size]
        if priority:
            filtered = filtered[filtered["priority"] == priority]

        if len(filtered) == 0:
            st.markdown(
                """
                <div class="no-result">
                    No exact match found.<br>
                    Try <strong>No Preference</strong> for fuel or adjust your budget.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            results = filtered.head(3)
            st.markdown('<div class="section-label" style="margin-top:8px;">Top Picks For You</div>', unsafe_allow_html=True)
            cols = st.columns(len(results))

            for idx, (col, (_, row)) in enumerate(zip(cols, results.iterrows())):
                img_path = get_car_image(row["car_name"])
                fuel_key = str(row["fuel"]).lower()
                fuel_badge_class = f"badge-{fuel_key}" if fuel_key in ["petrol", "diesel", "electric"] else "badge"
                fuel_icon = {"petrol": "⛽", "diesel": "🛢️", "electric": "⚡"}.get(fuel_key, "🚗")

                try:
                    rating_val = float(row["rating"])
                    full = int(rating_val)
                    half = 1 if rating_val - full >= 0.5 else 0
                    stars = "★" * full + ("½" if half else "") + "☆" * (5 - full - half)
                    rating_display = f"{stars} {rating_val:.1f}"
                except:
                    rating_display = str(row.get("rating", "N/A"))

                price_display = str(row["price"]) if "price" in row else "N/A"

                with col:
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="car-card">
                          <div class="car-img-container"><img src="{img_path}" alt="{row['car_name']}"/></div>
                          <div class="car-body">
                            {"<div style='background:#6366F1;color:white;padding:5px 12px;border-radius:999px;display:inline-block;font-size:10px;font-weight:700;margin-bottom:10px;'>🏆 TOP RECOMMENDATION</div>" if idx == 0 else ""}
                            <div class="car-brand">{row['brand']}</div>
                            <div class="car-name">{row['car_name']}</div>
                            <div class="car-price">{price_display}</div>
                            <div class="car-rating">{rating_display}</div>
                            <div class="car-desc">{row['description']}</div>
                            <div>
                              <span class="{fuel_badge_class}">{fuel_icon} {row['fuel'].upper()}</span>
                              <span class="badge">{str(row['budget']).replace('_',' ').title()}</span>
                              <span class="badge">{str(row['usage']).capitalize()}</span>
                              <span class="badge">{str(row['priority']).capitalize()}</span>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

st.markdown('<div class="divider" style="margin-top:2rem;"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">SmartCar AI &nbsp;&nbsp;•&nbsp;&nbsp; CarDekho Dataset &nbsp;&nbsp;•&nbsp;&nbsp; Linear Regression + Random Forest</div>',
    unsafe_allow_html=True
)
