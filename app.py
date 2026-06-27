# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartCar AI", page_icon="🚗", layout="wide")

# ── THEME ──────────────────────────────────────────────────────────────────────
page_bg    = "#0f1113"
card_bg    = "rgba(255,255,255,0.03)"
soft_yellow = "#E9C27A"
muted_text = "#BFBFBF"
accent     = soft_yellow

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

.stApp {{
    background-color: {page_bg};
    color: #EDEDED;
    font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif;
}}

/* Hide default streamlit menu/footer */
#MainMenu, footer {{ visibility: hidden; }}

/* Main container padding */
.block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 1200px; }}

/* ── HEADER ── */
.site-title {{
    font-family: 'Playfair Display', 'Times New Roman', serif;
    font-size: 42px;
    color: {soft_yellow};
    letter-spacing: 1px;
    margin: 0; padding: 0;
    line-height: 1.1;
}}
.site-subtitle {{
    color: {muted_text};
    font-size: 14px;
    margin-top: 4px;
    margin-bottom: 0;
}}
.nav-links {{
    text-align: right;
    color: #888;
    font-size: 13px;
    margin-top: 8px;
}}

/* ── DIVIDER ── */
hr {{ border-color: rgba(255,255,255,0.06); margin: 1rem 0; }}

/* ── CARDS ── */
.card {{
    background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    color: #EDEDED;
    height: 100%;
}}
.card-highlight {{
    border: 1.5px solid {accent};
    box-shadow: 0 8px 28px rgba(233,194,122,0.12);
}}
.card-title {{
    font-weight: 700;
    font-size: 15px;
    color: #fff;
    margin-bottom: 4px;
}}
.card-small {{
    color: {muted_text};
    font-size: 13px;
}}
.metric {{
    font-size: 22px;
    font-weight: 700;
    color: {accent};
    margin-top: 4px;
}}
.section-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: {muted_text};
    margin-bottom: 12px;
}}

/* ── STREAMLIT INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background: rgba(255,255,255,0.04) !important;
    color: #EDEDED !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}}
.stSelectbox > div > div {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #EDEDED !important;
}}
label, .stSlider label {{ color: {muted_text} !important; font-size: 13px !important; }}
.stSlider > div > div > div > div {{ background: {accent} !important; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {muted_text};
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(233,194,122,0.12) !important;
    color: {accent} !important;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: linear-gradient(90deg, {accent}, #dfc089) !important;
    color: #1a1208 !important;
    width: 100%;
    padding: 12px 18px;
    border-radius: 28px;
    font-weight: 700;
    font-size: 14px;
    border: none !important;
    box-shadow: 0 4px 16px rgba(233,194,122,0.25);
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.9; }}

/* ── RESULT CARDS ── */
.result-pill {{
    display: inline-block;
    background: rgba(233,194,122,0.1);
    border: 1px solid rgba(233,194,122,0.25);
    border-radius: 999px;
    padding: 3px 12px;
    color: {accent};
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
}}

/* ── CAR RECOMMENDER ── */
.car-card {{
    background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 12px;
}}
.car-name {{ font-weight: 700; font-size: 15px; color: #fff; }}
.car-brand {{ font-size: 12px; color: {muted_text}; margin-bottom: 6px; }}
.car-desc  {{ font-size: 13px; color: #ccc; line-height: 1.5; }}
.badge {{
    display: inline-block;
    background: rgba(233,194,122,0.1);
    border: 1px solid rgba(233,194,122,0.2);
    color: {accent};
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    margin-right: 5px;
    margin-top: 8px;
    font-weight: 500;
}}
.no-result {{
    color: {muted_text};
    font-size: 14px;
    padding: 24px;
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 12px;
    text-align: center;
}}

/* ── INFO BOX ── */
.stAlert {{ border-radius: 10px !important; }}
</style>
""", unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("<div class='site-title'>SmartCar AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='site-subtitle'>Used Car Price Prediction · ML-Powered Insights</div>", unsafe_allow_html=True)
with h2:
    st.markdown("<div class='nav-links' style='margin-top:16px;'>CarDekho Dataset &nbsp;·&nbsp; 2 Models</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── ML TRAINING ───────────────────────────────────────────────────────────────
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

    df.replace({'Fuel_Type':    {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    df.replace({'Seller_Type':  {'Dealer': 0, 'Individual': 1}},        inplace=True)
    df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}},         inplace=True)

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


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Price Predictor", "🚘  Car Recommender"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    lr_model, rf_model, lr_r2, lr_mae, rf_r2, rf_mae = load_and_train()

    # ── Model scores row ──
    st.markdown("<div class='section-label'>Model Performance</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    def score_card(col, model, label, value, highlight=False):
        border = f"border: 1.5px solid {accent};" if highlight else ""
        col.markdown(f"""
        <div class='card' style='padding:14px; text-align:center; {border}'>
            <div class='card-small'>{model}</div>
            <div class='card-small' style='font-size:11px; margin-top:2px;'>{label}</div>
            <div class='metric' style='font-size:18px;'>{value}</div>
        </div>""", unsafe_allow_html=True)

    score_card(s1, "Linear Regression", "R² Score",   f"{lr_r2:.3f}")
    score_card(s2, "Linear Regression", "Avg Error",  f"₹ {lr_mae:.2f}L")
    score_card(s3, "Random Forest",     "R² Score",   f"{rf_r2:.3f}", highlight=True)
    score_card(s4, "Random Forest",     "Avg Error",  f"₹ {rf_mae:.2f}L", highlight=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Inputs + results ──
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("<div class='section-label'>Car Details</div>", unsafe_allow_html=True)
        with st.form("price_form"):
            c1, c2 = st.columns(2)
            with c1:
                year           = st.selectbox("Year", list(range(2003, 2025)), index=12)
                present_price  = st.number_input("Showroom Price (Lakh ₹)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
                kms_driven     = st.number_input("Kilometres Driven", min_value=500, max_value=500000, value=30000, step=1000)
            with c2:
                fuel_type   = st.selectbox("Fuel Type",     ["Petrol", "Diesel", "CNG"])
                seller_type = st.selectbox("Seller Type",   ["Dealer", "Individual"])
                transmission= st.selectbox("Transmission",  ["Manual", "Automatic"])

            submitted = st.form_submit_button("⚡  GET PREDICTION")

    with right:
        st.markdown("<div class='section-label'>Predicted Price</div>", unsafe_allow_html=True)

        if submitted:
            fuel_map   = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            trans_map  = {"Manual": 0, "Automatic": 1}

            input_data = np.array([[
                year, present_price, kms_driven,
                fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]
            ]])

            lr_pred = max(0, lr_model.predict(input_data)[0])
            rf_pred = max(0, rf_model.predict(input_data)[0])
            better  = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"

            # LR card
            st.markdown(f"""
            <div class='card' style='margin-bottom:12px;'>
                <div class='card-small'>Linear Regression</div>
                <div class='metric'>₹ {lr_pred:.2f} Lakh</div>
            </div>""", unsafe_allow_html=True)

            # RF card (highlighted)
            st.markdown(f"""
            <div class='card card-highlight'>
                <div class='card-small'>Random Forest</div>
                <div class='metric'>₹ {rf_pred:.2f} Lakh</div>
                <div class='result-pill'>✦ More Accurate</div>
            </div>""", unsafe_allow_html=True)

            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by ₹{diff:.2f}L · {better} is the stronger predictor (R² {max(lr_r2, rf_r2):.3f})")
        else:
            st.markdown(f"""
            <div class='card' style='text-align:center; padding:40px 20px;'>
                <div style='font-size:32px;'>🔮</div>
                <div class='card-small' style='margin-top:10px;'>Fill in car details and hit<br><strong style='color:{accent};'>Get Prediction</strong></div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAR RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df_cars = load_car_data()

    st.markdown("<div class='section-label'>Find Your Perfect Car</div>", unsafe_allow_html=True)
    st.info("Set your preferences and we'll match you with the best options.")

    with st.form("reco_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            budget = st.selectbox("Budget", ["under_5L", "5-10L", "10-20L", "20L+"],
                format_func=lambda x: {"under_5L":"Under ₹5 Lakh","5-10L":"₹5–10 Lakh",
                                        "10-20L":"₹10–20 Lakh","20L+":"Above ₹20 Lakh"}[x])
            fuel = st.selectbox("Fuel", ["any","petrol","diesel","electric"],
                format_func=lambda x: {"any":"No Preference","petrol":"Petrol",
                                        "diesel":"Diesel","electric":"Electric"}[x])
        with c2:
            usage = st.selectbox("Usage", ["city","highway","both"],
                format_func=lambda x: {"city":"Mostly City","highway":"Mostly Highway","both":"Both"}[x])
            family_size = st.selectbox("Family Size", ["1-2","3-4","5+"],
                format_func=lambda x: {"1-2":"1–2 People","3-4":"3–4 People","5+":"5+ People"}[x])
        with c3:
            priority = st.selectbox("Priority", ["mileage","comfort","style","performance"],
                format_func=lambda x: {"mileage":"Best Mileage","comfort":"Comfort & Space",
                                        "style":"Stylish Looks","performance":"Performance"}[x])
            st.markdown("<br>", unsafe_allow_html=True)
            search = st.form_submit_button("🚘  Find My Car")

    if search:
        filtered = df_cars[df_cars["budget"] == budget].copy()
        if fuel != "any":
            filtered = filtered[filtered["fuel"] == fuel]
        filtered = filtered[(filtered["usage"] == usage) | (filtered["usage"] == "both")]
        filtered = filtered[filtered["family_size"] == family_size]

        priority_filtered = filtered[filtered["priority"] == priority]
        if len(priority_filtered) > 0:
            filtered = priority_filtered

        results = filtered.head(3)

        st.markdown("<div class='section-label' style='margin-top:16px;'>Top Picks For You</div>", unsafe_allow_html=True)

        if len(results) == 0:
            st.markdown("<div class='no-result'>No exact match found. Try selecting <strong>No Preference</strong> for fuel or adjusting your budget.</div>",
                unsafe_allow_html=True)
        else:
            cols = st.columns(len(results))
            for col, (_, row) in zip(cols, results.iterrows()):
                with col:
                    st.markdown(f"""
                    <div class='car-card'>
                        <div class='car-name'>{row['car_name']}</div>
                        <div class='car-brand'>{row['brand']}</div>
                        <div class='car-desc'>{row['description']}</div>
                        <div>
                            <span class='badge'>{row['budget'].replace('_',' ')}</span>
                            <span class='badge'>{row['fuel'].capitalize()}</span>
                            <span class='badge'>{row['usage'].capitalize()}</span>
                            <span class='badge'>{row['priority'].capitalize()}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; color:#555; font-size:12px;'>SmartCar AI &nbsp;·&nbsp; CarDekho Dataset &nbsp;·&nbsp; Linear Regression + Random Forest</div>",
    unsafe_allow_html=True)
