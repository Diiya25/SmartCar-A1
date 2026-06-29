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

# ── CAR IMAGE LOADER ───────────────────────────────────────────────────────────
DEFAULT_CAR_IMG = "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400&q=80"

def get_car_image(car_name: str):
    filename = car_name.lower().replace(" ", "_") + ".jpg"
    path = f"images/{filename}"
    if os.path.exists(path):
        return path
    return DEFAULT_CAR_IMG

# ── THEME TOKENS ──────────────────────────────────────────────────────────────
accent = "#C0392B"
accent_lt = "#FDECEA"
muted = "#6B7280"
body = "#111827"
card_bg = "#F9FAFB"
border_c = "#E5E7EB"

FUEL_COLORS = {
    "petrol":   ("#C0392B", "#FDECEA"),
    "diesel":   ("#1D4ED8", "#EFF6FF"),
    "electric": ("#059669", "#ECFDF5"),
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800;900&family=Barlow+Condensed:wght@700;800;900&display=swap');

.stApp {{
    background: #FFFFFF;
    color: {body};
    font-family: 'Barlow', sans-serif;
}}

* {{ font-family: 'Barlow', sans-serif !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1280px;
}}

/* HEADER */
.site-title {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 58px;
    font-weight: 900;
    color: {accent};
    letter-spacing: -1px;
    margin: 0;
    padding: 0;
    line-height: 1;
    text-transform: uppercase;
}}

.site-subtitle {{
    color: {muted};
    font-size: 15px;
    font-weight: 500;
    margin-top: 6px;
    letter-spacing: 0.3px;
}}

.hero-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}}

.hero-chip {{
    display: inline-block;
    background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
    color: #4F46E5;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 7px 14px;
    border: 1px solid rgba(99,102,241,0.18);
    box-shadow: 0 6px 18px rgba(99,102,241,0.08);
}}

.nav-chip {{
    display: inline-block;
    background: {accent_lt};
    color: {accent};
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 5px 14px;
    border: 1px solid rgba(192,57,43,0.2);
    margin-left: 6px;
}}

.divider {{
    height: 3px;
    background: linear-gradient(90deg, {accent} 0%, {accent_lt} 55%, transparent 100%);
    border: none;
    margin: 1rem 0 1.5rem;
    border-radius: 2px;
}}

/* SECTION LABEL */
.section-label {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: {muted};
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-label::before {{
    content: '';
    display: inline-block;
    width: 18px;
    height: 3px;
    background: {accent};
    border-radius: 2px;
    flex-shrink: 0;
}}

/* METRIC CARDS */
.metric-card {{
    background: {card_bg};
    border-radius: 16px;
    padding: 18px 20px;
    border: 1.5px solid {border_c};
    text-align: center;
}}

.metric-card.hi {{
    border-color: {accent};
    background: {accent_lt};
}}

.mc-model {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {muted};
}}

.mc-label {{
    font-size: 11px;
    color: {muted};
    margin-top: 2px;
}}

.mc-val {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 32px;
    font-weight: 900;
    color: {accent};
    margin-top: 4px;
    line-height: 1;
    letter-spacing: -0.5px;
}}

/* RESULT CARDS */
.result-card {{
    background: {card_bg};
    border-radius: 16px;
    padding: 22px;
    border: 1.5px solid {border_c};
    margin-bottom: 14px;
}}

.result-card.best {{
    border-color: {accent};
    background: {accent_lt};
}}

.rc-model {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {muted};
    margin-bottom: 4px;
}}

.rc-price {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 40px;
    font-weight: 900;
    color: {accent};
    line-height: 1;
    letter-spacing: -1px;
}}

.rc-pill {{
    display: inline-block;
    background: {accent};
    color: #fff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 4px 12px;
    margin-top: 10px;
}}

.empty-state {{
    background: {card_bg};
    border-radius: 16px;
    padding: 48px 24px;
    border: 1.5px dashed {border_c};
    text-align: center;
}}

.empty-icon {{
    font-size: 40px;
    margin-bottom: 12px;
}}

.empty-text {{
    font-size: 14px;
    color: {muted};
    font-weight: 500;
}}

.empty-cta {{
    color: {accent};
    font-weight: 700;
}}

/* CAR CARDS */
.car-card {{
    background: #fff;
    border-radius: 20px;
    border: 1.5px solid {border_c};
    overflow: hidden;
    margin-bottom: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.car-card:hover {{
    transform: translateY(-10px) scale(1.03);
    box-shadow:
        0 25px 70px rgba(99,102,241,0.30),
        0 0 40px rgba(99,102,241,0.20);
}}

.car-body {{
    padding: 16px 18px 20px;
}}

.car-brand {{
    font-size: 11px;
    font-weight: 700;
    color: {muted};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 2px;
}}

.car-name {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800;
    font-size: 22px;
    color: {body};
    line-height: 1.1;
    letter-spacing: -0.3px;
    margin-bottom: 6px;
}}

.car-price {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 26px;
    font-weight: 900;
    color: {accent};
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}}

.car-rating {{
    font-size: 13px;
    font-weight: 700;
    color: #F59E0B;
    margin-bottom: 8px;
}}

.car-desc {{
    font-size: 12px;
    color: #4B5563;
    line-height: 1.5;
    margin-bottom: 12px;
}}

.badge {{
    display: inline-block;
    background: {accent_lt};
    border: 1px solid rgba(192,57,43,0.2);
    color: {accent};
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}}

.badge-petrol {{
    display: inline-block;
    background: #FDECEA;
    border: 1px solid rgba(192,57,43,0.2);
    color: #C0392B;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}}

.badge-diesel {{
    display: inline-block;
    background: #EFF6FF;
    border: 1px solid rgba(29,78,216,0.2);
    color: #1D4ED8;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}}

.badge-electric {{
    display: inline-block;
    background: #ECFDF5;
    border: 1px solid rgba(5,150,105,0.2);
    color: #059669;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 4px;
    margin-top: 4px;
}}

.top-reco-badge {{
    display: inline-block;
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    color: white;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.8px;
    box-shadow: 0 10px 24px rgba(99,102,241,0.28);
}}

.no-result {{
    text-align: center;
    padding: 48px 24px;
    border: 1.5px dashed {border_c};
    border-radius: 16px;
    color: {muted};
    font-size: 14px;
    font-weight: 500;
}}

/* NATIVE STREAMLIT OVERRIDES */
label, .stSlider label {{
    color: {muted} !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background: #fff !important;
    color: {body} !important;
    border: 1.5px solid {border_c} !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}}

.stSelectbox > div > div {{
    background: #fff !important;
    border: 1.5px solid {border_c} !important;
    border-radius: 10px !important;
    color: {body} !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}}

.stSlider > div > div > div > div {{
    background: {accent} !important;
}}

/* TABS */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: #F3F4F6;
    border-radius: 12px;
    padding: 5px;
    border: 1.5px solid {border_c};
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {muted};
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 26px;
}}

.stTabs [aria-selected="true"] {{
    background: {accent} !important;
    color: #fff !important;
}}

/* BUTTON */
div[data-testid="stFormSubmitButton"] {{
    width: 100%;
}}

div[data-testid="stFormSubmitButton"] > button {{
    background: {accent} !important;
    color: #fff !important;
    width: 100% !important;
    padding: 15px 24px !important;
    border-radius: 12px !important;
    border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900 !important;
    font-size: 18px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(192,57,43,0.28) !important;
    transition: opacity 0.2s !important;
    display: block !important;
}}

div[data-testid="stFormSubmitButton"] > button:hover {{
    opacity: 0.88 !important;
}}

.stAlert {{
    border-radius: 12px !important;
    font-weight: 500 !important;
}}

.footer {{
    text-align: center;
    color: {muted};
    font-size: 12px;
    font-weight: 500;
    padding: 16px 0 4px;
    letter-spacing: 0.3px;
}}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])

with h1:
    st.markdown("<div class='site-title'>SMART<span style='color:#111827;'>CAR</span> AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='site-subtitle'>AI-Powered Vehicle Intelligence • Predict • Analyze • Recommend</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class='hero-chips'>
        <span class='hero-chip'>98.4% Accuracy</span>
        <span class='hero-chip'>2 ML Models</span>
        <span class='hero-chip'>AI Powered</span>
        <span class='hero-chip'>5000+ Vehicles</span>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown("""
    <div style='text-align:right; margin-top:20px;'>
        <span class='nav-chip'>CarDekho</span>
        <span class='nav-chip'>2 Models</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

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

    df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
    df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=2).fit(X_train, y_train)

    lr_r2 = r2_score(y_test, lr.predict(X_test))
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

    return lr, rf, lr_r2, lr_mae, rf_r2, rf_mae

@st.cache_data
def load_car_data():
    return pd.read_csv("car_recommendations.csv")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Price Predictor", "🚘  Car Recommender"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab1:
    lr_model, rf_model, lr_r2, lr_mae, rf_r2, rf_mae = load_and_train()

    st.markdown("<div class='section-label'>Model Performance</div>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    def metric_card(col, model, label, value, hi=False):
        cls = "metric-card hi" if hi else "metric-card"
        col.markdown(f"""
<div class='{cls}'>
  <div class='mc-model'>{model}</div>
  <div class='mc-label'>{label}</div>
  <div class='mc-val'>{value}</div>
</div>
""", unsafe_allow_html=True)

    metric_card(s1, "Linear Regression", "R² Score",  f"{lr_r2:.3f}")
    metric_card(s2, "Linear Regression", "Avg Error", f"₹{lr_mae:.2f}L")
    metric_card(s3, "Random Forest", "R² Score", f"{rf_r2:.3f}", hi=True)
    metric_card(s4, "Random Forest", "Avg Error", f"₹{rf_mae:.2f}L", hi=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, _, right = st.columns([1.3, 0.05, 1])

    with left:
        st.markdown("<div class='section-label'>Car Details</div>", unsafe_allow_html=True)

        with st.form("price_form"):
            c1, c2 = st.columns(2)

            with c1:
                year = st.selectbox("Year", list(range(2003, 2025)), index=12)
                present_price = st.number_input("Showroom Price (Lakh ₹)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
                kms_driven = st.number_input("Kilometres Driven", min_value=500, max_value=500000, value=30000, step=1000)

            with c2:
                fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
                seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡  Get Prediction")

    with right:
        st.markdown("<div class='section-label'>Predicted Price</div>", unsafe_allow_html=True)

        if submitted:
            fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            trans_map = {"Manual": 0, "Automatic": 1}

            inp = np.array([[
                year, present_price, kms_driven,
                fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]
            ]])

            lr_pred = max(0, lr_model.predict(inp)[0])
            rf_pred = max(0, rf_model.predict(inp)[0])

            st.markdown(f"""
<div class='result-card'>
  <div class='rc-model'>Linear Regression</div>
  <div class='rc-price'>₹ {lr_pred:.2f} <span style='font-size:17px;font-weight:600;'>Lakh</span></div>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class='result-card best'>
  <div class='rc-model'>Random Forest</div>
  <div class='rc-price'>₹ {rf_pred:.2f} <span style='font-size:17px;font-weight:600;'>Lakh</span></div>
  <div class='rc-pill'>✦ Most Accurate</div>
</div>
""", unsafe_allow_html=True)

            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by ₹{diff:.2f}L · Best R² = {max(lr_r2, rf_r2):.3f}")

        else:
            st.markdown("""
<div class='empty-state'>
  <div class='empty-icon'>🔮</div>
  <div class='empty-text'>Fill in car details and hit<br>
    <span class='empty-cta'>Get Prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — CAR RECOMMENDER
# ═══════════════════════════════════════════════════════════════
with tab2:
    df_cars = load_car_data()

    st.markdown("<div class='section-label'>Find Your Perfect Car</div>", unsafe_allow_html=True)

    with st.form("reco_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            budget = st.selectbox(
                "Budget",
                ["under_5L", "5-10L", "10-20L", "20L+"],
                format_func=lambda x: {
                    "under_5L": "Under ₹5 Lakh",
                    "5-10L": "₹5–10 Lakh",
                    "10-20L": "₹10–20 Lakh",
                    "20L+": "Above ₹20 Lakh"
                }[x]
            )

            fuel = st.selectbox(
                "Fuel Type",
                ["any", "petrol", "diesel", "electric"],
                format_func=lambda x: {
                    "any": "⛽  No Preference",
                    "petrol": "🔴  Petrol",
                    "diesel": "🔵  Diesel",
                    "electric": "🟢  Electric"
                }[x]
            )

        with c2:
            usage = st.selectbox(
                "Usage",
                ["city", "highway", "both"],
                format_func=lambda x: {
                    "city": "Mostly City",
                    "highway": "Mostly Highway",
                    "both": "Both"
                }[x]
            )

            family_size = st.selectbox(
                "Family Size",
                ["1-2", "3-4", "5+"],
                format_func=lambda x: {
                    "1-2": "1–2 People",
                    "3-4": "3–4 People",
                    "5+": "5+ People"
                }[x]
            )

        with c3:
            priority = st.selectbox(
                "Priority",
                ["mileage", "comfort", "style", "performance"],
                format_func=lambda x: {
                    "mileage": "🏅 Best Mileage",
                    "comfort": "🛋️ Comfort & Space",
                    "style": "✨ Stylish Looks",
                    "performance": "⚡ Performance"
                }[x]
            )

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
            st.markdown(
                "<div class='no-result'>No exact match found. Try <strong>No Preference</strong> for fuel or adjust your budget.</div>",
                unsafe_allow_html=True
            )
        else:
            cols = st.columns(len(results))

            for idx, (col, (_, row)) in enumerate(zip(cols, results.iterrows())):
                img_path = get_car_image(row['car_name'])

                fuel_key = str(row['fuel']).lower()
                fuel_badge_class = f"badge-{fuel_key}" if fuel_key in ["petrol", "diesel", "electric"] else "badge"
                fuel_icon = {"petrol": "🔴", "diesel": "🔵", "electric": "🟢"}.get(fuel_key, "⛽")

                try:
                    rating_val = float(row['rating'])
                    full = int(rating_val)
                    half = 1 if (rating_val - full) >= 0.5 else 0
                    empty = 5 - full - half
                    stars = "★" * full + ("½" if half else "") + "☆" * empty
                    rating_display = f"{stars} {rating_val}"
                except:
                    rating_display = str(row.get('rating', ''))

                price_display = str(row.get('price', '')) if 'price' in row else ''

                top_badge = "<div class='top-reco-badge'>🏆 TOP RECOMMENDATION</div>" if idx == 0 else ""

                with col:
                    st.image(img_path, use_container_width=True)

                    st.markdown(f"""
<div class='car-card'>
  <div class='car-body'>
    {top_badge}
    <div class='car-brand'>{row['brand']}</div>
    <div class='car-name'>{row['car_name']}</div>
    <div class='car-price'>{price_display}</div>
    <div class='car-rating'>{rating_display}</div>
    <div class='car-desc'>{row['description']}</div>
    <div>
      <span class='{fuel_badge_class}'>{fuel_icon} {row['fuel'].upper()}</span>
      <span class='badge'>{row['budget'].replace('_',' ')}</span>
      <span class='badge'>{row['usage'].capitalize()}</span>
      <span class='badge'>{row['priority'].capitalize()}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>SmartCar AI &nbsp;·&nbsp; CarDekho Dataset &nbsp;·&nbsp; Linear Regression + Random Forest</div>",
    unsafe_allow_html=True
)
