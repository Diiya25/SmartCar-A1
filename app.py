# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SmartCar AI",
    page_icon="🚗",
    layout="wide"
)

# ─── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title-box h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .title-box p  { color: #a0a0b0; font-size: 1rem; margin-top: 0.5rem; }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .result-card h3 { color: #a0a0b0; font-size: 0.95rem; margin-bottom: 0.4rem; }
    .result-card p  { color: #666688; font-size: 0.8rem; margin-top: 0.3rem; }
    .price-rf { color: #00d4aa; font-size: 2rem; font-weight: 700; }
    .price-lr { color: #e94560; font-size: 2rem; font-weight: 700; }
    .accuracy-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .accuracy-card .label { color: #a0a0b0; font-size: 0.8rem; }
    .accuracy-card .model-name { color: #ffffff; font-size: 1rem; font-weight: 600; margin-bottom: 0.2rem; }
    .accuracy-card .value { color: #e94560; font-size: 1.4rem; font-weight: 700; }
    .accuracy-card .value-green { color: #00d4aa; font-size: 1.4rem; font-weight: 700; }
    .section-title {
        color: #e94560;
        font-size: 1.2rem;
        font-weight: 600;
        border-left: 4px solid #e94560;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c73652);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
    }
    .winner-badge {
        background: #00d4aa22;
        border: 1px solid #00d4aa;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        color: #00d4aa;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .car-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        color: white;
    }
    .car-title { font-size: 1.3rem; font-weight: 700; color: #e94560; }
    .car-brand { font-size: 0.85rem; color: #aaaaaa; margin-bottom: 8px; }
    .car-desc  { font-size: 0.95rem; color: #dddddd; }
    .badge {
        display: inline-block;
        background-color: #e94560;
        color: white;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 8px;
    }
    .no-result {
        color: #e94560;
        font-size: 1.1rem;
        padding: 20px;
        border: 1px dashed #e94560;
        border-radius: 12px;
        text-align: center;
    }
    .info-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #a0a0b0;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🚗 SmartCar AI</h1>
    <p>Predict used car prices & find your perfect car match — powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💰 Price Predictor", "🔍 Car Recommender"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════
with tab1:

    @st.cache_resource
    def load_and_train():
        df = pd.read_csv("car_data.csv")

        # Filter out bikes — keep only cars (Present_Price > 2 lakh generally separates them)
        car_keywords = ['ritz','sx4','ciaz','wagon','swift','brezza','ertiga','dzire','alto',
                        'baleno','ignis','omni','fortuner','innova','corolla','etios','camry',
                        'land cruiser','i20','i10','grand','eon','xcent','creta','verna',
                        'elantra','city','brio','amaze','jazz']
        df = df[df['Car_Name'].str.lower().str.contains('|'.join(car_keywords), na=False)]

        # Drop Owner column (near-zero variance) and Car_Name (not useful for prediction)
        df = df.drop(['Car_Name', 'Owner'], axis=1)

        # Encode categorical columns
        df.replace({'Fuel_Type':   {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
        df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
        df.replace({'Transmission':{'Manual': 0, 'Automatic': 1}}, inplace=True)

        X = df.drop('Selling_Price', axis=1)
        Y = df['Selling_Price']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        lr_r2  = r2_score(Y_test, lr.predict(X_test))
        lr_mae = mean_absolute_error(Y_test, lr.predict(X_test))

        rf = RandomForestRegressor(n_estimators=100, random_state=2)
        rf.fit(X_train, Y_train)
        rf_r2  = r2_score(Y_test, rf.predict(X_test))
        rf_mae = mean_absolute_error(Y_test, rf.predict(X_test))

        return lr, rf, lr_r2, lr_mae, rf_r2, rf_mae

    lr_model, rf_model, lr_r2, lr_mae, rf_r2, rf_mae = load_and_train()

    # ── Accuracy Section ──────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 How Accurate Are Our Predictions?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        We trained two ML models on the CarDekho dataset to predict used car prices.
        R2 Score shows accuracy (closer to 1.0 = better). MAE shows average error in Lakh Rs.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="model-name">Basic Model</div>
            <div class="label">Linear Regression · R2 Score</div>
            <div class="value">{lr_r2:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="model-name">Basic Model</div>
            <div class="label">Linear Regression · Avg Error</div>
            <div class="value">{lr_mae:.2f} Lakh</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="model-name">Advanced Model</div>
            <div class="label">Random Forest · R2 Score</div>
            <div class="value-green">{rf_r2:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="model-name">Advanced Model</div>
            <div class="label">Random Forest · Avg Error</div>
            <div class="value-green">{rf_mae:.2f} Lakh</div>
        </div>""", unsafe_allow_html=True)

    # ── Input Section ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔧 Enter Your Car Details</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        year          = st.slider("📅 Year of Purchase", 2003, 2024, 2015)
        present_price = st.number_input("💵 Current Showroom Price (Lakh Rs)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
        kms_driven    = st.number_input("🛣️ Kilometres Driven", min_value=500, max_value=500000, value=30000, step=1000)

    with col_right:
        fuel_type    = st.selectbox("⛽ Fuel Type",    ["Petrol", "Diesel", "CNG"])
        seller_type  = st.selectbox("🏪 Seller Type",  ["Dealer", "Individual"])
        transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

    fuel_map   = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map  = {"Manual": 0, "Automatic": 1}

    input_data = np.array([[
        year, present_price, kms_driven,
        fuel_map[fuel_type], seller_map[seller_type],
        trans_map[transmission]
    ]])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Selling Price"):
        lr_pred = max(0, lr_model.predict(input_data)[0])
        rf_pred = max(0, rf_model.predict(input_data)[0])

        st.markdown('<div class="section-title">💡 Predicted Selling Price</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            st.markdown(f"""
            <div class="result-card">
                <h3>📈 Basic Model (Linear Regression)</h3>
                <div class="price-lr">₹ {lr_pred:.2f} Lakh</div>
                <p>Uses a straight-line formula to estimate price</p>
                <div style="color:#a0a0b0;font-size:0.8rem;margin-top:0.5rem;">Accuracy (R2): {lr_r2:.3f}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            badge = '<div class="winner-badge">✅ More Accurate</div>' if rf_r2 > lr_r2 else ''
            st.markdown(f"""
            <div class="result-card">
                <h3>🌳 Advanced Model (Random Forest)</h3>
                <div class="price-rf">₹ {rf_pred:.2f} Lakh</div>
                <p>Uses 100 decision trees to estimate price</p>
                {badge}
                <div style="color:#a0a0b0;font-size:0.8rem;margin-top:0.5rem;">Accuracy (R2): {rf_r2:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        diff = abs(rf_pred - lr_pred)
        better = "Advanced Model (Random Forest)" if rf_r2 > lr_r2 else "Basic Model (Linear Regression)"
        st.info(f"📌 The two models differ by ₹{diff:.2f} Lakh. {better} has a higher accuracy score ({max(lr_r2, rf_r2):.3f}), making it the more reliable estimate.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — CAR RECOMMENDER
# ══════════════════════════════════════════════════════════════════════
with tab2:

    @st.cache_data
    def load_car_data():
        return pd.read_csv("car_recommendations.csv")

    df_cars = load_car_data()

    st.markdown('<div class="section-title">🔍 Find Your Perfect Car</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Not sure which car to buy? Answer a few simple questions and our recommender will suggest the best car for your needs and budget!
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        budget = st.selectbox("💰 What's your budget?", [
            "under_5L", "5-10L", "10-20L", "20L+"
        ], format_func=lambda x: {
            "under_5L": "Under ₹5 Lakh",
            "5-10L": "₹5 – 10 Lakh",
            "10-20L": "₹10 – 20 Lakh",
            "20L+": "Above ₹20 Lakh"
        }[x])

        fuel = st.selectbox("⛽ Fuel preference?", [
            "any", "petrol", "diesel", "electric"
        ], format_func=lambda x: {
            "any": "No Preference",
            "petrol": "Petrol",
            "diesel": "Diesel",
            "electric": "Electric"
        }[x])

    with col2:
        usage = st.selectbox("🛣️ How will you use it?", [
            "city", "highway", "both"
        ], format_func=lambda x: {
            "city": "Mostly City Driving",
            "highway": "Mostly Highway",
            "both": "Both City & Highway"
        }[x])

        family_size = st.selectbox("👨‍👩‍👧 Family size?", [
            "1-2", "3-4", "5+"
        ], format_func=lambda x: {
            "1-2": "1–2 People",
            "3-4": "3–4 People",
            "5+": "5 or More People"
        }[x])

    priority = st.selectbox("⭐ What matters most to you?", [
        "mileage", "comfort", "style", "performance"
    ], format_func=lambda x: {
        "mileage": "Best Mileage",
        "comfort": "Comfort & Space",
        "style": "Stylish Looks",
        "performance": "Performance & Power"
    }[x])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Find My Car"):
        filtered = df_cars.copy()
        filtered = filtered[filtered["budget"] == budget]

        if fuel != "any":
            filtered = filtered[filtered["fuel"] == fuel]

        filtered = filtered[
            (filtered["usage"] == usage) | (filtered["usage"] == "both")
        ]
        filtered = filtered[filtered["family_size"] == family_size]

        priority_filtered = filtered[filtered["priority"] == priority]
        if len(priority_filtered) > 0:
            filtered = priority_filtered

        results = filtered.head(3)

        st.markdown('<div class="section-title">🎯 Top Picks For You</div>', unsafe_allow_html=True)

        if len(results) == 0:
            st.markdown("""
                <div class='no-result'>
                    😕 No exact match found for your combination.<br>
                    Try changing your fuel preference to <b>No Preference</b> or adjusting your budget.
                </div>
            """, unsafe_allow_html=True)
        else:
            for i, row in results.iterrows():
                st.markdown(f"""
                    <div class='car-card'>
                        <div class='car-title'>🚘 {row['car_name']}</div>
                        <div class='car-brand'>Brand: {row['brand']}</div>
                        <div class='car-desc'>{row['description']}</div>
                        <div>
                            <span class='badge'>💰 {row['budget'].replace('_', ' ')}</span>
                            <span class='badge'>⛽ {row['fuel'].capitalize()}</span>
                            <span class='badge'>🛣️ {row['usage'].capitalize()}</span>
                            <span class='badge'>⭐ {row['priority'].capitalize()}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Recommendations based on your preferences. Prices may vary by city and dealer.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<center><span style='color:#555;font-size:0.8rem;'>SmartCar AI · Summer Training Project · UGI Prayagraj · CarDekho Dataset</span></center>", unsafe_allow_html=True)
