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

st.set_page_config(page_title="SmartCar AI", page_icon="\U0001f697", layout="wide")

DEFAULT_CAR_IMG = "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400&q=80"

def get_car_image(car_name: str):
    filename = car_name.lower().replace(" ", "_") + ".jpg"
    path = f"images/{filename}"
    if os.path.exists(path):
        return path
    return DEFAULT_CAR_IMG

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800;900&family=Barlow+Condensed:wght@700;800;900&display=swap');
.stApp { background: #FFFFFF; color: #111827; font-family: 'Barlow', sans-serif; }
* { font-family: 'Barlow', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0rem 2.5rem 3rem 2.5rem; max-width: 1280px; }
.hero {
    position: relative; width: 100%; min-height: 320px;
    background: radial-gradient(ellipse at 60% 50%, rgba(192,57,43,0.18) 0%, transparent 70%),
                radial-gradient(ellipse at 20% 50%, rgba(192,57,43,0.08) 0%, transparent 60%),
                linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
    border-bottom: 1px solid rgba(192,57,43,0.2);
    display: flex; align-items: center;
    padding: 2rem 2rem 1.5rem; overflow: hidden;
    margin: 0 -2.5rem 2rem -2.5rem;
}
.hero-left { flex: 1; z-index: 2; }
.hero-tag {
    display: inline-block; background: rgba(192,57,43,0.15);
    border: 1px solid rgba(192,57,43,0.4); color: #E74C3C;
    font-size: 11px; font-weight: 800; letter-spacing: 2.5px;
    text-transform: uppercase; border-radius: 999px; padding: 5px 16px; margin-bottom: 16px;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 72px; font-weight: 900; color: #FFFFFF;
    letter-spacing: -2px; line-height: 0.9;
    text-transform: uppercase; margin-bottom: 6px;
}
.hero-title span { color: #E74C3C; }
.hero-subtitle { color: #6B7280; font-size: 14px; font-weight: 500; margin-top: 14px; }
.hero-chips { margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
.hero-chip {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    color: #9CA3AF; font-size: 11px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase; border-radius: 999px; padding: 5px 14px;
}
.hero-car-wrap {
    flex: 0 0 480px; display: flex; align-items: center;
    justify-content: center; position: relative; z-index: 2;
}
.hero-car-glow {
    position: absolute; width: 360px; height: 180px;
    background: radial-gradient(ellipse, rgba(192,57,43,0.35) 0%, transparent 70%);
    bottom: -20px; left: 50%; transform: translateX(-50%);
    filter: blur(20px); animation: glowPulse 3s ease-in-out infinite;
}
@keyframes glowPulse {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scaleX(1); }
    50%       { opacity: 1;   transform: translateX(-50%) scaleX(1.15); }
}
.hero-car-img {
    width: 440px;
    filter: drop-shadow(0 20px 60px rgba(192,57,43,0.3));
    animation: floatCar 4s ease-in-out infinite;
    transform-origin: center bottom;
}
@keyframes floatCar {
    0%, 100% { transform: translateY(0px) rotate(-1deg); }
    50%       { transform: translateY(-14px) rotate(1deg); }
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, #E74C3C 0%, rgba(231,76,60,0.3) 50%, transparent 100%);
    border: none; margin: 1.5rem 0; border-radius: 2px;
}
.section-label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 13px; font-weight: 800; text-transform: uppercase;
    letter-spacing: 2px; color: #6B7280; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::before {
    content: ''; display: inline-block; width: 18px; height: 3px;
    background: #E74C3C; border-radius: 2px; flex-shrink: 0;
}
.metric-card {
    background: rgba(255,255,255,0.04); backdrop-filter: blur(12px);
    border-radius: 16px; padding: 18px 20px;
    border: 1px solid rgba(255,255,255,0.08); text-align: center;
}
.metric-card.hi {
    border-color: rgba(231,76,60,0.4); background: rgba(192,57,43,0.1);
    box-shadow: 0 0 24px rgba(192,57,43,0.15);
}
.mc-model { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #6B7280; }
.mc-label { font-size: 11px; color: #6B7280; margin-top: 2px; }
.mc-val {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 32px; font-weight: 900; color: #E74C3C;
    margin-top: 4px; line-height: 1; letter-spacing: -0.5px;
}
.result-card {
    background: rgba(255,255,255,0.04); backdrop-filter: blur(12px);
    border-radius: 16px; padding: 22px;
    border: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;
}
.result-card.best {
    border-color: rgba(231,76,60,0.5); background: rgba(192,57,43,0.12);
    box-shadow: 0 0 32px rgba(192,57,43,0.2);
}
.rc-model { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #6B7280; margin-bottom: 4px; }
.rc-price {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 40px; font-weight: 900; color: #E74C3C; line-height: 1; letter-spacing: -1px;
}
.rc-pill {
    display: inline-block; background: #E74C3C; color: #fff;
    font-size: 10px; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; border-radius: 999px; padding: 4px 12px; margin-top: 10px;
}
.empty-state {
    background: rgba(255,255,255,0.03); border-radius: 16px;
    padding: 48px 24px; border: 1px dashed rgba(255,255,255,0.1); text-align: center;
}
.empty-icon { font-size: 40px; margin-bottom: 12px; }
.empty-text { font-size: 14px; color: #6B7280; font-weight: 500; }
.empty-cta { color: #E74C3C; font-weight: 700; }
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
.car-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(16px);
    border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);
    overflow: hidden; margin-bottom: 8px;
    animation: fadeSlideUp 0.5s ease forwards;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.car-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(192,57,43,0.25), 0 0 0 1px rgba(231,76,60,0.3);
}
.car-img-container { overflow: hidden; border-radius: 12px 12px 0 0; background: rgba(255,255,255,0.03); }
.car-img-container img { width: 100%; transition: transform 0.5s ease; }
.car-img-container:hover img { transform: scale(1.07); }
.car-body { padding: 16px 18px 20px; }
.car-brand { font-size: 11px; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
.car-name {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800; font-size: 22px; color: #FFFFFF;
    line-height: 1.1; letter-spacing: -0.3px; margin-bottom: 6px;
}
.car-price {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 28px; font-weight: 900; color: #E74C3C;
    letter-spacing: -0.5px; margin-bottom: 4px;
}
.car-rating { font-size: 13px; font-weight: 700; color: #F59E0B; margin-bottom: 8px; }
.car-desc { font-size: 12px; color: #9CA3AF; line-height: 1.5; margin-bottom: 12px; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12); color: #9CA3AF;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.badge-petrol {
    display: inline-block; background: rgba(192,57,43,0.15);
    border: 1px solid rgba(192,57,43,0.35); color: #E74C3C;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.badge-diesel {
    display: inline-block; background: rgba(29,78,216,0.15);
    border: 1px solid rgba(29,78,216,0.35); color: #60A5FA;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.badge-electric {
    display: inline-block; background: rgba(5,150,105,0.15);
    border: 1px solid rgba(5,150,105,0.35); color: #34D399;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.no-result {
    text-align: center; padding: 48px 24px;
    border: 1px dashed rgba(255,255,255,0.1); border-radius: 16px;
    color: #6B7280; font-size: 14px; font-weight: 500;
}
label, .stSlider label {
    color: #6B7280 !important; font-size: 12px !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important; color: #F0F0F0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; font-size: 15px !important; font-weight: 600 !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #F0F0F0 !important;
    font-size: 15px !important; font-weight: 600 !important;
}
.stSlider > div > div > div > div { background: #E74C3C !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.04);
    border-radius: 12px; padding: 5px; border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6B7280; border-radius: 8px;
    font-size: 14px; font-weight: 700; padding: 10px 26px;
}
.stTabs [aria-selected="true"] {
    background: #E74C3C !important; color: #fff !important;
    box-shadow: 0 0 16px rgba(231,76,60,0.4) !important;
}
div[data-testid="stFormSubmitButton"] { width: 100%; }
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%) !important;
    color: #fff !important; width: 100% !important; padding: 15px 24px !important;
    border-radius: 12px !important; border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900 !important; font-size: 18px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(192,57,43,0.45) !important;
    transition: opacity 0.2s, box-shadow 0.2s !important; display: block !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.9 !important; box-shadow: 0 6px 32px rgba(192,57,43,0.65) !important;
}
.stAlert { border-radius: 12px !important; font-weight: 500 !important; }
.footer { text-align: center; color: #4B5563; font-size: 12px; font-weight: 500; padding: 16px 0 4px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    df = pd.read_csv("car_data.csv")
    car_keywords = [
        'ritz','sx4','ciaz','wagon','swift','brezza','ertiga','dzire','alto',
        'baleno','ignis','omni','fortuner','innova','corolla','etios','camry',
        'land cruiser','i20','i10','grand','eon','xcent','creta','verna',
        'elantra','city','brio','amaze','jazz'
    ]
    df = df[df['Car_Name'].str.lower().str.contains('|'.join(car_keywords), na=False)]
    df = df.drop(['Car_Name', 'Owner'], axis=1)
    df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
    df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=2).fit(X_train, y_train)
    lr_r2  = r2_score(y_test, lr.predict(X_test))
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    rf_r2  = r2_score(y_test, rf.predict(X_test))
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
    return lr, rf, lr_r2, lr_mae, rf_r2, rf_mae

@st.cache_data
def load_car_data():
    return pd.read_csv("car_recommendations.csv")

hero_img = "https://imgd.aeplcdn.com/664x374/n/cw/ec/44709/fortuner-exterior-right-front-three-quarter-2.jpeg?q=80"
st.markdown(f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-tag">\U0001f916 AI · ML Powered</div>
    <div class="hero-title">Smart<span>Car</span><br>AI</div>
    <div class="hero-subtitle">Used Car Price Prediction &nbsp;·&nbsp; Personalised Recommendations</div>
    <div class="hero-chips">
      <span class="hero-chip">CarDekho Dataset</span>
      <span class="hero-chip">2 ML Models</span>
      <span class="hero-chip">Linear Regression</span>
      <span class="hero-chip">Random Forest</span>
    </div>
  </div>
  <div class="hero-car-wrap">
    <div class="hero-car-glow"></div>
    <img class="hero-car-img" src="{hero_img}" alt="SmartCar Hero"/>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["\U0001f52e  Price Predictor", "\U0001f698  Car Recommender"])

with tab1:
    lr_model, rf_model, lr_r2, lr_mae, rf_r2, rf_mae = load_and_train()

    st.markdown("<div class='section-label'>Model Performance</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    def metric_card(col, model, label, value, hi=False):
        cls = "metric-card hi" if hi else "metric-card"
        col.markdown(f"<div class='{cls}'><div class='mc-model'>{model}</div><div class='mc-label'>{label}</div><div class='mc-val'>{value}</div></div>", unsafe_allow_html=True)

    metric_card(s1, "Linear Regression", "R² Score",  f"{lr_r2:.3f}")
    metric_card(s2, "Linear Regression", "Avg Error", f"₹{lr_mae:.2f}L")
    metric_card(s3, "Random Forest",     "R² Score",  f"{rf_r2:.3f}", hi=True)
    metric_card(s4, "Random Forest",     "Avg Error", f"₹{rf_mae:.2f}L", hi=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, _, right = st.columns([1.3, 0.05, 1])

    with left:
        st.markdown("<div class='section-label'>Car Details</div>", unsafe_allow_html=True)
        with st.form("price_form"):
            c1, c2 = st.columns(2)
            with c1:
                year          = st.selectbox("Year", list(range(2003, 2025)), index=12)
                present_price = st.number_input("Showroom Price (Lakh ₹)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
                kms_driven    = st.number_input("Kilometres Driven", min_value=500, max_value=500000, value=30000, step=1000)
            with c2:
                fuel_type    = st.selectbox("Fuel Type",    ["Petrol", "Diesel", "CNG"])
                seller_type  = st.selectbox("Seller Type",  ["Dealer", "Individual"])
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡  Get Prediction")

    with right:
        st.markdown("<div class='section-label'>Predicted Price</div>", unsafe_allow_html=True)
        if submitted:
            fuel_map   = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            trans_map  = {"Manual": 0, "Automatic": 1}
            inp = np.array([[year, present_price, kms_driven,
                             fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]]])
            lr_pred = max(0, lr_model.predict(inp)[0])
            rf_pred = max(0, rf_model.predict(inp)[0])
            st.markdown(f"<div class='result-card'><div class='rc-model'>Linear Regression</div><div class='rc-price'>₹ {lr_pred:.2f} <span style='font-size:17px;font-weight:600;'>Lakh</span></div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-card best'><div class='rc-model'>Random Forest</div><div class='rc-price'>₹ {rf_pred:.2f} <span style='font-size:17px;font-weight:600;'>Lakh</span></div><div class='rc-pill'>✶ Most Accurate</div></div>", unsafe_allow_html=True)
            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by ₹{diff:.2f}L · Best R² = {max(lr_r2, rf_r2):.3f}")
        else:
            st.markdown("<div class='empty-state'><div class='empty-icon'>🔮</div><div class='empty-text'>Fill in car details and hit<br><span class='empty-cta'>Get Prediction</span></div></div>", unsafe_allow_html=True)

with tab2:
    df_cars = load_car_data()
    st.markdown("<div class='section-label'>Find Your Perfect Car</div>", unsafe_allow_html=True)

    with st.form("reco_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            budget = st.selectbox("Budget", ["under_5L", "5-10L", "10-20L", "20L+"],
                format_func=lambda x: {"under_5L": "Under ₹5 Lakh", "5-10L": "₹5–10 Lakh",
                                        "10-20L": "₹10–20 Lakh", "20L+": "Above ₹20 Lakh"}[x])
            fuel = st.selectbox("Fuel Type", ["any", "petrol", "diesel", "electric"],
                format_func=lambda x: {"any": "⛽  No Preference", "petrol": "🔴  Petrol",
                                        "diesel": "🔵  Diesel", "electric": "🟢  Electric"}[x])
        with c2:
            usage = st.selectbox("Usage", ["city", "highway", "both"],
                format_func=lambda x: {"city": "Mostly City", "highway": "Mostly Highway", "both": "Both"}[x])
            family_size = st.selectbox("Family Size", ["1-2", "3-4", "5+"],
                format_func=lambda x: {"1-2": "1–2 People", "3-4": "3–4 People", "5+": "5+ People"}[x])
        with c3:
            priority = st.selectbox("Priority", ["mileage", "comfort", "style", "performance"],
                format_func=lambda x: {"mileage": "🏅 Best Mileage", "comfort": "🛋️ Comfort & Space",
                                        "style": "✨ Stylish Looks", "performance": "⚡ Performance"}[x])
            st.markdown("<br>", unsafe_allow_html=True)
            search = st.form_submit_button("🚘  Find My Car")

    if search:
        filtered = df_cars[df_cars["budget"] == budget].copy()
        if fuel != "any":
            filtered = filtered[filtered["fuel"] == fuel]
        filtered = filtered[(filtered["usage"] == usage) | (filtered["usage"] == "both")]
        filtered = filtered[filtered["family_size"] == family_size]
        pf = filtered[filtered["priority"] == priority]
        if len(pf) > 0:
            filtered = pf
        results = filtered.head(3)

        st.markdown("<div class='section-label' style='margin-top:8px;'>Top Picks For You</div>", unsafe_allow_html=True)

        if len(results) == 0:
            st.markdown("<div class='no-result'>No exact match found. Try <strong>No Preference</strong> for fuel or adjust your budget.</div>", unsafe_allow_html=True)
        else:
            cols = st.columns(len(results))
            for col, (_, row) in zip(cols, results.iterrows()):
                img_path = get_car_image(row['car_name'])
                fuel_key = str(row['fuel']).lower()
                fuel_badge_class = f"badge-{fuel_key}" if fuel_key in ["petrol","diesel","electric"] else "badge"
                fuel_icon = {"petrol": "🔴", "diesel": "🔵", "electric": "🟢"}.get(fuel_key, "⛽")
                try:
                    rating_val = float(row['rating'])
                    full = int(rating_val); half = 1 if (rating_val - full) >= 0.5 else 0
                    stars = "★" * full + ("½" if half else "") + "☆" * (5 - full - half)
                    rating_display = f"{stars} {rating_val}"
                except:
                    rating_display = str(row.get('rating', ''))
                price_display = str(row.get('price', '')) if 'price' in row else ''
                with col:
                    if os.path.exists(str(img_path)):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.markdown(f"<div class='car-img-container'><img src='{img_path}' alt='{row[chr(99)+chr(97)+chr(114)+chr(95)+chr(110)+chr(97)+chr(109)+chr(101)]}'/></div>", unsafe_allow_html=True)
                    st.markdown(f"""<div class='car-card'><div class='car-body'>
<div class='car-brand'>{row['brand']}</div>
<div class='car-name'>{row['car_name']}</div>
<div class='car-price'>{price_display}</div>
<div class='car-rating'>{rating_display}</div>
<div class='car-desc'>{row['description']}</div>
<div><span class='{fuel_badge_class}'>{fuel_icon} {row['fuel'].upper()}</span>
<span class='badge'>{row['budget'].replace('_',' ')}</span>
<span class='badge'>{row['usage'].capitalize()}</span>
<span class='badge'>{row['priority'].capitalize()}</span></div>
</div></div>""", unsafe_allow_html=True)

st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>SmartCar AI · CarDekho Dataset · Linear Regression + Random Forest</div>", unsafe_allow_html=True)
