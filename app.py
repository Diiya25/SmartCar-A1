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
from car_3d_component import render_3d_car
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
}

/* animated grid floor */
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(37,99,235,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(37,99,235,0.05) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.25) 40%, transparent 100%);
    pointer-events: none;
}

/* floating particles */
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background-image:
        radial-gradient(1.5px 1.5px at 20% 30%, rgba(6,182,212,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 60% 20%, rgba(37,99,235,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 60%, rgba(6,182,212,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 70%, rgba(37,99,235,0.3) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 90% 40%, rgba(6,182,212,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 80%, rgba(37,99,235,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 85%, rgba(6,182,212,0.3) 0%, transparent 100%);
    pointer-events: none;
    animation: particleDrift 8s ease-in-out infinite alternate;
}
@keyframes particleDrift {
    0%   { transform: translateY(0px); }
    100% { transform: translateY(-20px); }
}

.hero-content { position: relative; z-index: 2; max-width: 680px; }

.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.3);
    color: #60A5FA; font-size: 11px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    border-radius: 999px; padding: 6px 18px; margin-bottom: 28px;
}
.hero-eyebrow-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #06B6D4;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; } 50% { opacity: 0.2; }
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 80px; font-weight: 400; color: #0f172a;
    letter-spacing: 3px; line-height: 0.92;
    text-transform: uppercase; margin-bottom: 0;
}
.hero-title .accent { color: #2563EB; }
.hero-title .accent2 { color: #06B6D4; }

.hero-tagline {
    color: #475569; font-size: 16px; font-weight: 400;
    margin-top: 20px; line-height: 1.6; max-width: 520px;
    letter-spacing: 0.2px;
}

/* ── HERO STATS ── */
.hero-stats {
    display: flex; gap: 32px; margin-top: 36px; flex-wrap: wrap;
}
.hero-stat {
    display: flex; flex-direction: column; gap: 2px;
}
.hero-stat-num {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 36px; color: #0f172a; letter-spacing: 1px; line-height: 1;
}
.hero-stat-num span { color: #2563EB; }
.hero-stat-label {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.hero-stat-divider {
    width: 1px; background: rgba(37,99,235,0.2); align-self: stretch; margin: 4px 0;
}

.hero-cta {
    margin-top: 36px; display: flex; gap: 14px; align-items: center;
}
.hero-badge {
    background: #fff; border: 1.5px solid rgba(37,99,235,0.2);
    color: #1d4ed8; font-size: 11px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    border-radius: 999px; padding: 6px 16px;
}

/* ── SCROLL INDICATOR ── */
.scroll-hint {
    position: absolute; bottom: 32px; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    color: #94a3b8; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    animation: bobUp 2s ease-in-out infinite;
    z-index: 2;
}
@keyframes bobUp {
    0%,100% { transform: translateX(-50%) translateY(0); }
    50%      { transform: translateX(-50%) translateY(-8px); }
}
.scroll-arrow {
    width: 24px; height: 24px;
    border-right: 2px solid #94a3b8;
    border-bottom: 2px solid #94a3b8;
    transform: rotate(45deg);
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.5) 30%, rgba(6,182,212,0.4) 60%, transparent);
    border: none; margin: 2rem 0;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 13px; letter-spacing: 4px;
    color: #94a3b8; margin-bottom: 18px;
    display: flex; align-items: center; gap: 12px;
    text-transform: uppercase;
}
.section-label::before {
    content: ''; display: inline-block; width: 22px; height: 2px;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    border-radius: 2px; flex-shrink: 0;
}

/* ── GLASSMORPHISM CARDS ── */
.glass-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1.5px solid rgba(37,99,235,0.12);
    border-radius: 20px; padding: 22px;
}
.glass-card.glow {
    border-color: rgba(37,99,235,0.3);
    box-shadow: 0 0 32px rgba(37,99,235,0.12), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── METRIC CARDS ── */
.metric-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(20px);
    border-radius: 18px; padding: 22px 20px;
    border: 1.5px solid rgba(37,99,235,0.12);
    text-align: center;
    box-shadow: 0 4px 20px rgba(37,99,235,0.06);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    border-color: rgba(37,99,235,0.3);
    box-shadow: 0 0 24px rgba(37,99,235,0.12);
}
.metric-card.hi {
    border-color: rgba(6,182,212,0.3);
    background: rgba(6,182,212,0.05);
    box-shadow: 0 0 28px rgba(6,182,212,0.10);
}
.mc-model { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; }
.mc-label { font-size: 11px; color: #64748b; margin-top: 3px; }
.mc-val {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 38px; color: #2563EB;
    margin-top: 8px; line-height: 1; letter-spacing: 1px;
}
.metric-card.hi .mc-val { color: #06B6D4; }

/* ── FORM INPUTS ── */
label, .stSlider label {
    color: #64748b !important; font-size: 11px !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.85) !important; color: #0f172a !important;
    border: 1.5px solid rgba(37,99,235,0.15) !important;
    border-radius: 10px !important; font-size: 15px !important; font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid rgba(37,99,235,0.15) !important;
    border-radius: 10px !important; color: #0f172a !important;
    font-size: 15px !important; font-weight: 500 !important;
}
.stSlider > div > div > div > div { background: #2563EB !important; }

/* ── HEALTH SCORE ── */
.health-wrap {
    background: rgba(6,182,212,0.06);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 16px; padding: 20px; margin-bottom: 16px;
}
.health-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; margin-bottom: 4px; }
.health-score {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 48px; color: #06B6D4; line-height: 1; letter-spacing: 2px;
}
.health-bar-bg { background: rgba(37,99,235,0.1); border-radius: 999px; height: 6px; margin-top: 10px; overflow: hidden; }
.health-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2563EB, #06B6D4); transition: width 1s ease; }

/* ── RESULT CARDS ── */
.result-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(20px);
    border-radius: 16px; padding: 22px;
    border: 1.5px solid rgba(37,99,235,0.12); margin-bottom: 14px;
    box-shadow: 0 4px 16px rgba(37,99,235,0.06);
}
.result-card.best {
    border-color: rgba(6,182,212,0.35);
    background: rgba(6,182,212,0.06);
    box-shadow: 0 0 32px rgba(6,182,212,0.12);
}
.rc-model { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; margin-bottom: 6px; }
.rc-price {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 44px; color: #2563EB; line-height: 1; letter-spacing: 1px;
}
.result-card.best .rc-price { color: #06B6D4; }
.rc-pill {
    display: inline-block; background: linear-gradient(135deg, #2563EB, #06B6D4); color: #fff;
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; border-radius: 999px;
    padding: 5px 14px; margin-top: 10px;
}

/* ── WHY THIS PRICE ── */
.why-card {
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 14px; padding: 18px 20px; margin-top: 14px;
}
.why-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.8px; color: #94a3b8; margin-bottom: 12px; }
.why-item { font-size: 13px; color: #475569; padding: 5px 0; border-bottom: 1px solid rgba(37,99,235,0.06); }
.why-item:last-child { border-bottom: none; }

/* ── EMPTY STATE ── */
.empty-state {
    background: rgba(255,255,255,0.6); border-radius: 18px;
    padding: 56px 24px; border: 1.5px dashed rgba(37,99,235,0.2); text-align: center;
}
.empty-icon { font-size: 44px; margin-bottom: 14px; }
.empty-text { font-size: 14px; color: #64748b; font-weight: 500; }
.empty-cta { color: #2563EB; font-weight: 700; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.6);
    border-radius: 14px; padding: 5px;
    border: 1.5px solid rgba(37,99,235,0.12);
    box-shadow: 0 2px 12px rgba(37,99,235,0.05);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748b; border-radius: 10px;
    font-size: 14px; font-weight: 700; padding: 11px 28px; letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563EB, #0ea5e9) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}

/* ── BUTTON ── */
div[data-testid="stFormSubmitButton"] { width: 100%; }
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%) !important;
    color: #fff !important; width: 100% !important; padding: 16px 24px !important;
    border-radius: 12px !important; border: none !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-weight: 400 !important; font-size: 20px !important;
    letter-spacing: 3px !important; text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.4) !important;
    transition: all 0.25s !important; display: block !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 8px 40px rgba(37,99,235,0.6) !important;
    transform: translateY(-2px) !important;
}

/* ── CAR CARDS ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
.car-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(20px);
    border-radius: 22px; border: 1.5px solid rgba(37,99,235,0.12);
    overflow: hidden; animation: fadeSlideUp 0.5s ease forwards;
    box-shadow: 0 4px 24px rgba(37,99,235,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s;
}
.car-card:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0 25px 70px rgba(37,99,235,0.25), 0 0 40px rgba(6,182,212,0.15);
    border-color: rgba(37,99,235,0.3);
}
.car-img-container {
    overflow: hidden; height: 185px;
    background: linear-gradient(135deg, #dbeafe, #e0f2fe);
    display: flex; align-items: center; justify-content: center;
    position: relative;
}
.car-img-container::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 60px;
    background: linear-gradient(transparent, rgba(5,5,5,0.8));
    pointer-events: none;
}
.car-img-container img {
    width: 100%; height: 100%; object-fit: cover;
    transition: transform 0.5s ease;
    filter: brightness(0.92) saturate(1.1);
}
.car-img-container:hover img { transform: scale(1.07); }

.top-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2563EB, #06B6D4);
    color: #fff; font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    border-radius: 999px; padding: 5px 14px; margin-bottom: 12px;
}
.car-body { padding: 18px 18px 20px; }
.car-brand { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 3px; }
.car-name {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 26px; color: #0f172a;
    line-height: 1.05; letter-spacing: 1px; margin-bottom: 6px;
}
.car-price {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 20px; color: #2563EB;
    letter-spacing: 0.5px; margin-bottom: 4px;
}
.car-rating { font-size: 13px; font-weight: 700; color: #F59E0B; margin-bottom: 8px; }
.car-desc { font-size: 12px; color: #64748b; line-height: 1.6; margin-bottom: 12px; }

.badge {
    display: inline-block; background: rgba(37,99,235,0.07);
    border: 1px solid rgba(37,99,235,0.15); color: #3b82f6;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.badge-petrol {
    display: inline-block; background: rgba(37,99,235,0.1);
    border: 1px solid rgba(37,99,235,0.3); color: #1d4ed8;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.badge-diesel {
    display: inline-block; background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.3); color: #0284c7;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.badge-electric {
    display: inline-block; background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3); color: #059669;
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}
.no-result {
    text-align: center; padding: 52px 24px;
    border: 1.5px dashed rgba(37,99,235,0.2); border-radius: 18px;
    background: rgba(255,255,255,0.5);
    color: #94a3b8; font-size: 14px; font-weight: 500;
}
.stAlert { border-radius: 12px !important; font-weight: 500 !important; }
.footer {
    text-align: center; color: #94a3b8; font-size: 11px;
    font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 24px 0 8px;
}
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
        <div class="hero-stat-num">57</div>
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


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  ...all your existing hero html...
</div>
""", unsafe_allow_html=True)

from car_3d_component import render_3d_car
render_3d_car("assets/chevrolet_corvette_c5_blue.glb", height=460)

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
            # ── SCANNING ANIMATION ──
            with st.spinner("🤖 AI scanning vehicle data..."):
                time.sleep(1.5)

            fuel_map   = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            trans_map  = {"Manual": 0, "Automatic": 1}
            inp = np.array([[year, present_price, kms_driven,
                             fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]]])
            lr_pred = max(0, lr_model.predict(inp)[0])
            rf_pred = max(0, rf_model.predict(inp)[0])

            # ── VEHICLE HEALTH SCORE ──
            health_score = min(99, max(55, int(100 - (kms_driven / 8000) - (2024 - year) * 1.5)))
            health_color = "#06B6D4" if health_score >= 80 else "#F59E0B" if health_score >= 60 else "#EF4444"
            st.markdown(f"""
<div class='health-wrap'>
  <div class='health-label'>Vehicle Health Score</div>
  <div class='health-score' style='color:{health_color};'>{health_score}<span style='font-size:22px;'>%</span></div>
  <div class='health-bar-bg'>
    <div class='health-bar-fill' style='width:{health_score}%; background: linear-gradient(90deg, #2563EB, {health_color});'></div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── PRICE RESULTS ──
            st.markdown(f"""<div class='result-card'>
  <div class='rc-model'>Linear Regression</div>
  <div class='rc-price'>₹ {lr_pred:.2f} <span style='font-size:17px;font-weight:600;font-family:Inter,sans-serif;'>Lakh</span></div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class='result-card best'>
  <div class='rc-model'>Random Forest</div>
  <div class='rc-price'>₹ {rf_pred:.2f} <span style='font-size:17px;font-weight:600;font-family:Inter,sans-serif;'>Lakh</span></div>
  <div class='rc-pill'>✦ Most Accurate</div>
</div>""", unsafe_allow_html=True)

            # ── WHY THIS PRICE ──
            reasons = []
            if kms_driven < 30000:  reasons.append("✅ Very low mileage — high resale value")
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

            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by ₹{diff:.2f}L · Best R² = {max(lr_r2, rf_r2):.3f}")
        else:
            st.markdown("""<div class='empty-state'>
  <div class='empty-icon'>🔮</div>
  <div class='empty-text'>Fill in car details and hit<br>
    <span class='empty-cta'>Analyse Vehicle</span>
  </div>
</div>""", unsafe_allow_html=True)

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
                img_path = get_car_image(row['car_name'])
                fuel_key = str(row['fuel']).lower()
                fuel_badge_class = f"badge-{fuel_key}" if fuel_key in ["petrol","diesel","electric"] else "badge"
                fuel_icon = {"petrol": "🔵", "diesel": "🔷", "electric": "🟢"}.get(fuel_key, "⛽")
                try:
                    rating_val = float(row['rating'])
                    full = int(rating_val)
                    half = 1 if (rating_val - full) >= 0.5 else 0
                    stars = "★" * full + ("½" if half else "") + "☆" * (5 - full - half)
                    rating_display = f"{stars} {rating_val}"
                except:
                    rating_display = str(row.get('rating', ''))
                price_display = str(row.get('price', '')) if 'price' in row else ''
                top_badge = "<div class='top-badge'>🏆 Top Recommendation</div>" if idx == 0 else ""

                with col:
                    if os.path.exists(str(img_path)):
                        st.image(img_path, use_container_width=True)
                        st.markdown(f"""<div class='car-card'><div class='car-body'>
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
</div></div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class='car-card'>
<div class='car-img-container'><img src='{img_path}' alt='{row["car_name"]}'/></div>
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
</div></div></div>""", unsafe_allow_html=True)

st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>SmartCar AI &nbsp;·&nbsp; CarDekho Dataset &nbsp;·&nbsp; Linear Regression + Random Forest</div>", unsafe_allow_html=True)
