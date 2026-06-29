# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartCar AI", page_icon="🚗", layout="wide")

DEFAULT_CAR_IMG = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Toyota_Corolla_E210_India_2019_front.jpg/320px-Toyota_Corolla_E210_India_2019_front.jpg"

def get_car_image(car_name: str):
    filename = car_name.lower().replace(" ", "_") + ".jpg"
    path = f"images/{filename}"
    if os.path.exists(path):
        return path
    return DEFAULT_CAR_IMG

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Bebas+Neue&display=swap');

/* ── BASE ── */
.stApp { background: #E8F4FD; color: #0f172a; font-family: 'Inter', sans-serif; }
* { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 3rem; max-width: 1280px; }

/* ── HERO ── */
.hero {
    position: relative; width: 100%; min-height: 88vh;
    background:
        radial-gradient(ellipse at 30% 50%, rgba(37,99,235,0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 30%, rgba(6,182,212,0.08) 0%, transparent 50%),
        linear-gradient(160deg, #dbeafe 0%, #e8f4fd 40%, #eff6ff 100%);
    display: flex; align-items: center;
    padding: 0 3rem;
    margin: 0 -2.5rem 0 -2.5rem;
    overflow: hidden;
    border-bottom: 1.5px solid rgba(37,99,235,0.12);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(37,99,235,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(37,99,235,0.05) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.3) 40%, transparent 100%);
    pointer-events: none;
}
.hero-content { position: relative; z-index: 2; max-width: 680px; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(37,99,235,0.1);
    border: 1.5px solid rgba(37,99,235,0.25);
    color: #1d4ed8; font-size: 11px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    border-radius: 999px; padding: 6px 18px; margin-bottom: 28px;
}
.hero-eyebrow-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #0ea5e9;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }
.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 86px; font-weight: 400; color: #0f172a;
    letter-spacing: 3px; line-height: 0.88;
    text-transform: uppercase; margin-bottom: 0;
}
.hero-title .accent  { color: #2563eb; }
.hero-title .accent2 { color: #0ea5e9; }
.hero-tagline {
    color: #475569; font-size: 15px; font-weight: 400;
    margin-top: 20px; line-height: 1.65; max-width: 520px;
}
.hero-stats { display: flex; gap: 32px; margin-top: 36px; flex-wrap: wrap; }
.hero-stat { display: flex; flex-direction: column; gap: 2px; }
.hero-stat-num {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 36px; color: #0f172a; letter-spacing: 1px; line-height: 1;
}
.hero-stat-num span { color: #2563eb; }
.hero-stat-label { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; }
.hero-stat-divider { width: 1px; background: rgba(37,99,235,0.2); align-self: stretch; margin: 4px 0; }
.hero-cta { margin-top: 36px; display: flex; gap: 12px; flex-wrap: wrap; }
.hero-badge {
    background: #fff; border: 1.5px solid rgba(37,99,235,0.2);
    color: #1d4ed8; font-size: 11px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    border-radius: 999px; padding: 6px 16px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
}
.scroll-hint {
    position: absolute; bottom: 32px; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    color: #94a3b8; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    animation: bobUp 2s ease-in-out infinite; z-index: 2;
}
@keyframes bobUp { 0%,100% { transform: translateX(-50%) translateY(0); } 50% { transform: translateX(-50%) translateY(-8px); } }
.scroll-arrow {
    width: 22px; height: 22px;
    border-right: 2px solid #94a3b8; border-bottom: 2px solid #94a3b8;
    transform: rotate(45deg);
}

/* ── DIVIDER ── */
.divider {
    height: 1.5px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.3) 30%, rgba(14,165,233,0.3) 60%, transparent);
    border: none; margin: 2rem 0;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 13px; letter-spacing: 4px; color: #94a3b8;
    margin-bottom: 18px; display: flex; align-items: center; gap: 12px; text-transform: uppercase;
}
.section-label::before {
    content: ''; display: inline-block; width: 22px; height: 2px;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    border-radius: 2px; flex-shrink: 0;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(16px);
    border-radius: 18px; padding: 20px;
    border: 1.5px solid rgba(37,99,235,0.1);
    text-align: center;
    box-shadow: 0 4px 20px rgba(37,99,235,0.06);
    transition: box-shadow 0.25s, border-color 0.25s;
}
.metric-card:hover { box-shadow: 0 8px 32px rgba(37,99,235,0.12); border-color: rgba(37,99,235,0.25); }
.metric-card.hi {
    border-color: rgba(14,165,233,0.35);
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(224,242,254,0.8));
    box-shadow: 0 4px 24px rgba(14,165,233,0.15);
}
.mc-model { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; }
.mc-label { font-size: 11px; color: #64748b; margin-top: 3px; }
.mc-val { font-family: 'Bebas Neue', sans-serif !important; font-size: 38px; color: #2563eb; margin-top: 8px; line-height: 1; letter-spacing: 1px; }
.metric-card.hi .mc-val { color: #0284c7; }

/* ── FORM INPUTS ── */
label, .stSlider label {
    color: #64748b !important; font-size: 11px !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.85) !important; color: #0f172a !important;
    border: 1.5px solid rgba(37,99,235,0.15) !important;
    border-radius: 10px !important; font-size: 15px !important; font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.05) !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid rgba(37,99,235,0.15) !important;
    border-radius: 10px !important; color: #0f172a !important;
    font-size: 15px !important; font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.05) !important;
}
.stSlider > div > div > div > div { background: #2563eb !important; }

/* ── HEALTH SCORE ── */
.health-wrap {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(224,242,254,0.7));
    border: 1.5px solid rgba(14,165,233,0.25);
    border-radius: 16px; padding: 20px; margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(14,165,233,0.1);
}
.health-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; margin-bottom: 4px; }
.health-score { font-family: 'Bebas Neue', sans-serif !important; font-size: 52px; line-height: 1; letter-spacing: 2px; }
.health-bar-bg { background: rgba(37,99,235,0.1); border-radius: 999px; height: 6px; margin-top: 10px; overflow: hidden; }
.health-bar-fill { height: 100%; border-radius: 999px; transition: width 1s ease; }

/* ── RESULT CARDS ── */
.result-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(16px);
    border-radius: 16px; padding: 22px;
    border: 1.5px solid rgba(37,99,235,0.1); margin-bottom: 14px;
    box-shadow: 0 4px 16px rgba(37,99,235,0.06);
}
.result-card.best {
    border-color: rgba(14,165,233,0.4);
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(224,242,254,0.8));
    box-shadow: 0 6px 32px rgba(14,165,233,0.15);
}
.rc-model { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; margin-bottom: 6px; }
.rc-price { font-family: 'Bebas Neue', sans-serif !important; font-size: 44px; color: #2563eb; line-height: 1; letter-spacing: 1px; }
.result-card.best .rc-price { color: #0284c7; }
.rc-pill {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    color: #fff; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; border-radius: 999px; padding: 5px 14px; margin-top: 10px;
}

/* ── WHY CARD ── */
.why-card {
    background: rgba(255,255,255,0.7);
    border: 1.5px solid rgba(37,99,235,0.12);
    border-radius: 14px; padding: 18px 20px; margin-top: 14px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.05);
}
.why-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; margin-bottom: 12px; }
.why-item { font-size: 13px; color: #475569; padding: 5px 0; border-bottom: 1px solid rgba(37,99,235,0.06); }
.why-item:last-child { border-bottom: none; }

/* ── EMPTY STATE ── */
.empty-state {
    background: rgba(255,255,255,0.6); border-radius: 18px;
    padding: 56px 24px; border: 1.5px dashed rgba(37,99,235,0.2); text-align: center;
}
.empty-icon { font-size: 44px; margin-bottom: 14px; }
.empty-text { font-size: 14px; color: #64748b; font-weight: 500; }
.empty-cta { color: #2563eb; font-weight: 700; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.6);
    border-radius: 14px; padding: 5px;
    border: 1.5px solid rgba(37,99,235,0.12);
    box-shadow: 0 2px 12px rgba(37,99,235,0.05);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748b; border-radius: 10px;
    font-size: 14px; font-weight: 700; padding: 11px 28px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.3) !important;
}

/* ── BUTTON ── */
div[data-testid="stFormSubmitButton"] { width: 100%; }
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
    color: #fff !important; width: 100% !important; padding: 16px 24px !important;
    border-radius: 12px !important; border: none !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-weight: 400 !important; font-size: 20px !important;
    letter-spacing: 3px !important; text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
    transition: all 0.25s !important; display: block !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 8px 36px rgba(37,99,235,0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── CAR CARDS ── */
@keyframes fadeSlideUp { from { opacity:0; transform:translateY(28px); } to { opacity:1; transform:translateY(0); } }
.car-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(20px);
    border-radius: 22px; border: 1.5px solid rgba(37,99,235,0.1);
    overflow: hidden; animation: fadeSlideUp 0.45s ease forwards;
    box-shadow: 0 4px 24px rgba(37,99,235,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s;
}
.car-card:hover {
    transform: translateY(-8px) scale(1.01);
    box-shadow: 0 20px 60px rgba(37,99,235,0.18), 0 0 0 1.5px rgba(14,165,233,0.3);
    border-color: rgba(37,99,235,0.25);
}
.car-img-container {
    overflow: hidden; height: 185px;
    background: linear-gradient(135deg, #dbeafe, #e0f2fe);
    position: relative;
}
.car-img-container::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 50px;
    background: linear-gradient(transparent, rgba(232,244,253,0.6));
    pointer-events: none;
}
.car-img-container img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.5s ease; }
.car-img-container:hover img { transform: scale(1.06); }

/* ── TOP BADGE ── */
.top-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    color: #fff; font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    border-radius: 999px; padding: 5px 14px; margin-bottom: 10px;
}
.car-price-tag{
    font-family:'Bebas Neue',sans-serif !important;
    font-size:22px;
    color:#2563eb;
    font-weight:700;
    white-space:nowrap;
}
.car-body { padding: 18px 18px 20px; }
.car-brand { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 3px; }
.car-name { font-family: 'Bebas Neue', sans-serif !important; font-size: 26px; color: #0f172a; line-height: 1.05; letter-spacing: 1px; margin-bottom: 4px; }

/* ── PRICE + RATING ROW ── */
.car-meta-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap; gap: 6px; }
.car-price-tag {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 24px; color: #2563eb; letter-spacing: 0.5px; line-height: 1;
}
.car-rating-wrap { display: flex; align-items: center; gap: 6px; }
.star-bar { display: flex; gap: 2px; }
.star { font-size: 13px; }
.star.full  { color: #f59e0b; }
.star.half  { color: #fbbf24; }
.star.empty { color: #cbd5e1; }
.rating-num { font-size: 13px; font-weight: 700; color: #475569; }

.car-desc { font-size: 12px; color: #64748b; line-height: 1.6; margin-bottom: 12px; }

/* ── BADGES ── */
.badge { display:inline-block; background:rgba(37,99,235,0.07); border:1px solid rgba(37,99,235,0.15); color:#3b82f6; border-radius:999px; padding:3px 10px; font-size:10px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; margin-right:4px; margin-top:4px; }
.badge-petrol  { display:inline-block; background:rgba(37,99,235,0.08); border:1px solid rgba(37,99,235,0.25); color:#1d4ed8; border-radius:999px; padding:3px 10px; font-size:10px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; margin-right:4px; margin-top:4px; }
.badge-diesel  { display:inline-block; background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.25); color:#0284c7; border-radius:999px; padding:3px 10px; font-size:10px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; margin-right:4px; margin-top:4px; }
.badge-electric{ display:inline-block; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); color:#059669; border-radius:999px; padding:3px 10px; font-size:10px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; margin-right:4px; margin-top:4px; }

.no-result { text-align:center; padding:52px 24px; border:1.5px dashed rgba(37,99,235,0.2); border-radius:18px; color:#94a3b8; font-size:14px; font-weight:500; background:rgba(255,255,255,0.5); }
.stAlert { border-radius:12px !important; font-weight:500 !important; }
.footer { text-align:center; color:#94a3b8; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; padding:24px 0 8px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-eyebrow">
      <div class="hero-eyebrow-dot"></div>
      AI · ML Powered &nbsp;·&nbsp; Live
    </div>
    <div class="hero-title">
      Smart<span class="accent">Car</span><br>
      <span class="accent2">AI</span>
    </div>
    <div class="hero-tagline">
      AI-powered vehicle intelligence — predict prices, analyse health,<br>
      and find your perfect car with machine learning.
    </div>
    <div class="hero-stats">
      <div class="hero-stat">
        <div class="hero-stat-num">98<span>%</span></div>
        <div class="hero-stat-label">Model Accuracy</div>
      </div>
      <div class="hero-stat-divider"></div>
      <div class="hero-stat">
        <div class="hero-stat-num">2</div>
        <div class="hero-stat-label">ML Models</div>
      </div>
      <div class="hero-stat-divider"></div>
      <div class="hero-stat">
        <div class="hero-stat-num">250<span>+</span></div>
        <div class="hero-stat-label">Cars Analysed</div>
      </div>
      <div class="hero-stat-divider"></div>
      <div class="hero-stat">
        <div class="hero-stat-num">101</div>
        <div class="hero-stat-label">Recommendations</div>
      </div>
    </div>
    <div class="hero-cta">
      <span class="hero-badge">⚡ Linear Regression</span>
      <span class="hero-badge">🌲 Random Forest</span>
      <span class="hero-badge">📊 CarDekho Dataset</span>
    </div>
  </div>
  <div class="scroll-hint">
    <div class="scroll-arrow"></div>
    Scroll
  </div>
</div>
""", unsafe_allow_html=True)

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
    lr_r2  = r2_score(y_test, lr.predict(X_test))
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    rf_r2  = r2_score(y_test, rf.predict(X_test))
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
    return lr, rf, lr_r2, lr_mae, rf_r2, rf_mae

@st.cache_data
def load_car_data():
    return pd.read_csv("car_recommendations.csv")

def render_stars(rating_val):
    try:
        val = float(rating_val)
        full  = int(val)
        half  = 1 if (val - full) >= 0.5 else 0
        empty = 5 - full - half
        stars_html = ""
        for _ in range(full):  stars_html += "<span class='star full'>★</span>"
        if half:               stars_html += "<span class='star half'>★</span>"
        for _ in range(empty): stars_html += "<span class='star empty'>★</span>"
        return f"<div class='car-rating-wrap'><div class='star-bar'>{stars_html}</div><span class='rating-num'>{val}</span></div>"
    except:
        return ""

# ── TABS ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🔮  Price Predictor", "🚘  Car Recommender"])

# ═══════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTOR
# ═══════════════════════════════════════════════════════
with tab1:
    lr_model, rf_model, lr_r2, lr_mae, rf_r2, rf_mae = load_and_train()

    st.markdown("<div class='section-label'>Model Performance</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    def metric_card(col, model, label, value, hi=False):
        cls = "metric-card hi" if hi else "metric-card"
        col.markdown(f"""<div class='{cls}'>
  <div class='mc-model'>{model}</div>
  <div class='mc-label'>{label}</div>
  <div class='mc-val'>{value}</div>
</div>""", unsafe_allow_html=True)

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
            submitted = st.form_submit_button("⚡  Analyse Vehicle")

    with right:
        st.markdown("<div class='section-label'>Predicted Price</div>", unsafe_allow_html=True)
        if submitted:
            with st.spinner("🤖 AI scanning vehicle data..."):
                time.sleep(1.5)
            fuel_map   = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            trans_map  = {"Manual": 0, "Automatic": 1}
            inp = np.array([[year, present_price, kms_driven,
                             fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]]])
            lr_pred = max(0, lr_model.predict(inp)[0])
            rf_pred = max(0, rf_model.predict(inp)[0])

            health_score = min(99, max(55, int(100 - (kms_driven / 8000) - (2024 - year) * 1.5)))
            health_color = "#0284c7" if health_score >= 80 else "#f59e0b" if health_score >= 60 else "#ef4444"
            bar_gradient = f"linear-gradient(90deg, #2563eb, {health_color})"

            st.markdown(f"""<div class='health-wrap'>
  <div class='health-label'>Vehicle Health Score</div>
  <div class='health-score' style='color:{health_color};'>{health_score}<span style='font-size:22px;font-family:Inter,sans-serif;'>%</span></div>
  <div class='health-bar-bg'>
    <div class='health-bar-fill' style='width:{health_score}%; background:{bar_gradient};'></div>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class='result-card'>
  <div class='rc-model'>Linear Regression</div>
  <div class='rc-price'>₹ {lr_pred:.2f} <span style='font-size:16px;font-weight:600;font-family:Inter,sans-serif;'>Lakh</span></div>
</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class='result-card best'>
  <div class='rc-model'>Random Forest</div>
  <div class='rc-price'>₹ {rf_pred:.2f} <span style='font-size:16px;font-weight:600;font-family:Inter,sans-serif;'>Lakh</span></div>
  <div class='rc-pill'>✦ Most Accurate</div>
</div>""", unsafe_allow_html=True)

            reasons = []
            if kms_driven < 30000:   reasons.append("✅ Very low mileage — high resale value")
            elif kms_driven < 60000: reasons.append("✅ Low mileage increases value")
            else:                    reasons.append("⚠️ High mileage reduces value")
            if year >= 2020:         reasons.append("✅ Recent model year — premium pricing")
            elif year >= 2017:       reasons.append("✅ Newer model year")
            else:                    reasons.append("⚠️ Older model year reduces value")
            if transmission == "Automatic": reasons.append("✅ Automatic transmission commands premium")
            if seller_type == "Dealer":     reasons.append("✅ Dealer-maintained — higher trust value")
            if fuel_type == "Diesel":       reasons.append("✅ Diesel — better highway mileage premium")

            reasons_html = "".join([f"<div class='why-item'>{r}</div>" for r in reasons])
            st.markdown(f"""<div class='why-card'>
  <div class='why-title'>Why This Price?</div>
  {reasons_html}
</div>""", unsafe_allow_html=True)
            st.info(f"Models differ by ₹{abs(rf_pred - lr_pred):.2f}L · Best R² = {max(lr_r2, rf_r2):.3f}")
        else:
            st.markdown("""<div class='empty-state'>
  <div class='empty-icon'>🔮</div>
  <div class='empty-text'>Fill in car details and hit<br>
    <span class='empty-cta'>Analyse Vehicle</span></div>
</div>""", unsafe_allow_html=True)

def render_stars(rating):
    rating = float(rating)

    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half

    stars = ""

    stars += "★" * full
    stars += "☆" * empty

    return f"""
    <div style='display:flex;align-items:center;gap:6px;margin-bottom:8px;'>
        <span style='color:#f59e0b;font-size:16px;'>{stars}</span>
        <span style='font-size:13px;font-weight:600;color:#475569;'>{rating}</span>
    </div>
    """

# ═══════════════════════════════════════════════════════
# TAB 2 — CAR RECOMMENDER
# ═══════════════════════════════════════════════════════
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
                format_func=lambda x: {"any": "⛽  No Preference", "petrol": "🔵  Petrol",
                                        "diesel": "🔷  Diesel", "electric": "🟢  Electric"}[x])
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
        with st.spinner("🤖 AI matching your perfect car..."):
            time.sleep(1.2)

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
            st.markdown("<div class='no-result'>No exact match found.<br>Try <strong>No Preference</strong> for fuel or adjust your budget.</div>", unsafe_allow_html=True)
        else:
            cols = st.columns(len(results))
            for idx, (col, (_, row)) in enumerate(zip(cols, results.iterrows())):
                img_path  = get_car_image(row['car_name'])
                fuel_key  = str(row['fuel']).lower()
                fuel_cls  = f"badge-{fuel_key}" if fuel_key in ["petrol","diesel","electric"] else "badge"
                fuel_icon = {"petrol": "🔵", "diesel": "🔷", "electric": "🟢"}.get(fuel_key, "⛽")
                top_badge = "<div class=\'top-badge\'>🏆 Top Recommendation</div>" if idx == 0 else ""

                # ── price ──
                price_val = str(row.get("price", "")).strip()
                price_html = f"<div class=\'car-price-tag\'>{price_val}</div>" if price_val else ""

                # ── stars ──
                rating_html = render_stars(row.get("rating", ""))

                with col:
                    st.markdown(f"""
<div class="car-card">
  <div class="car-img-container">
    <img src="{img_path}" alt="{row['car_name']}" style="width:100%;height:185px;object-fit:cover;"/>
  </div>
  <div class="car-body">
    {top_badge}
    <div style="display:flex;justify-content:space-between;align-items:center;">
    <div class="car-name">{row["car_name"]}</div>
    <div class="car-price-tag">{row["price"]}</div>
</div>
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div class="car-name">{row["car_name"]}</div>
    <div class="car-price-tag">{row["price"]}</div>
</div>

{rating_html}
    <div class="car-desc">{row["description"]}</div>
    <div>
      <span class="{fuel_cls}">{fuel_icon} {row["fuel"].upper()}</span>
      <span class="badge">{row["budget"].replace("_"," ")}</span>
      <span class="badge">{row["usage"].capitalize()}</span>
      <span class="badge">{row["priority"].capitalize()}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>SmartCar AI &nbsp;·&nbsp; CarDekho Dataset &nbsp;·&nbsp; Linear Regression + Random Forest</div>", unsafe_allow_html=True)
