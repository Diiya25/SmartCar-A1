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

# ── CAR IMAGE MAP ──────────────────────────────────────────────────────────────
CAR_IMAGES = {
    "Maruti Swift":      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/2018_Maruti_Suzuki_Swift_ZXI%2B_AT_%28front%29.jpg/320px-2018_Maruti_Suzuki_Swift_ZXI%2B_AT_%28front%29.jpg",
    "Maruti Alto":       "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/2019_Maruti_Suzuki_Alto_800_LXI_%28facelift%2C_front%29_in_Kolkata.jpg/320px-2019_Maruti_Suzuki_Alto_800_LXI_%28facelift%2C_front%29_in_Kolkata.jpg",
    "Maruti Baleno":     "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Maruti_Suzuki_Baleno_%28second_generation%2C_facelift%29%2C_front_8.19.22.jpg/320px-Maruti_Suzuki_Baleno_%28second_generation%2C_facelift%29%2C_front_8.19.22.jpg",
    "Maruti Dzire":      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/2022_Maruti_Suzuki_Dzire_ZXI_Plus_AT_%28front%29.jpg/320px-2022_Maruti_Suzuki_Dzire_ZXI_Plus_AT_%28front%29.jpg",
    "Maruti Wagon R":    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/2019_Maruti_Suzuki_Wagon_R_VXI%2B_AGS_%28front%29.jpg/320px-2019_Maruti_Suzuki_Wagon_R_VXI%2B_AGS_%28front%29.jpg",
    "Maruti Ertiga":     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/2022_Maruti_Suzuki_Ertiga_ZXI%2B_AT_%28front%29.jpg/320px-2022_Maruti_Suzuki_Ertiga_ZXI%2B_AT_%28front%29.jpg",
    "Maruti Brezza":     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/2022_Maruti_Suzuki_Brezza_ZXI%2B_AT_%28front%29.jpg/320px-2022_Maruti_Suzuki_Brezza_ZXI%2B_AT_%28front%29.jpg",
    "Maruti Ciaz":       "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Maruti_Suzuki_Ciaz_Alpha_2019_%28front%29.jpg/320px-Maruti_Suzuki_Ciaz_Alpha_2019_%28front%29.jpg",
    "Maruti Ignis":      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/2020_Maruti_Suzuki_Ignis_Zeta_AMT_%28front%29.jpg/320px-2020_Maruti_Suzuki_Ignis_Zeta_AMT_%28front%29.jpg",
    "Hyundai i10":       "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Hyundai_Grand_i10_Nios_Sportz_AMT_%28front%29%2C_Kolkata_2020.jpg/320px-Hyundai_Grand_i10_Nios_Sportz_AMT_%28front%29%2C_Kolkata_2020.jpg",
    "Hyundai i20":       "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/2020_Hyundai_i20_Asta_%28O%29_1.0_Turbo_DCT_%28front%29.jpg/320px-2020_Hyundai_i20_Asta_%28O%29_1.0_Turbo_DCT_%28front%29.jpg",
    "Hyundai Creta":     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Hyundai_Creta_SX_%28O%29_Turbo_DCT_2023_%28front%29.jpg/320px-Hyundai_Creta_SX_%28O%29_Turbo_DCT_2023_%28front%29.jpg",
    "Hyundai Verna":     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/2023_Hyundai_Verna_SX_%28O%29_Turbo_DCT_%28front%29.jpg/320px-2023_Hyundai_Verna_SX_%28O%29_Turbo_DCT_%28front%29.jpg",
    "Hyundai Xcent":     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Hyundai_Xcent_S_CRDi_%28facelift%29_front.jpg/320px-Hyundai_Xcent_S_CRDi_%28facelift%29_front.jpg",
    "Hyundai Elantra":   "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/2019_Hyundai_Elantra_SX%28O%29_1.5_CRDi_AT_%28front%29.jpg/320px-2019_Hyundai_Elantra_SX%28O%29_1.5_CRDi_AT_%28front%29.jpg",
    "Hyundai Grand i10": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Hyundai_Grand_i10_Nios_Sportz_AMT_%28front%29%2C_Kolkata_2020.jpg/320px-Hyundai_Grand_i10_Nios_Sportz_AMT_%28front%29%2C_Kolkata_2020.jpg",
    "Honda City":        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/2020_Honda_City_ZX_CVT_%28front%29.jpg/320px-2020_Honda_City_ZX_CVT_%28front%29.jpg",
    "Honda Amaze":       "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/2021_Honda_Amaze_V_CVT_%28front%29.jpg/320px-2021_Honda_Amaze_V_CVT_%28front%29.jpg",
    "Honda Jazz":        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/2020_Honda_Jazz_VX_CVT_%28front%29.jpg/320px-2020_Honda_Jazz_VX_CVT_%28front%29.jpg",
    "Honda Brio":        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Honda_Brio_S_MT_%28facelift%2C_front%29%2C_Kolkata_2016.jpg/320px-Honda_Brio_S_MT_%28facelift%2C_front%29%2C_Kolkata_2016.jpg",
    "Toyota Innova":     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Toyota_Innova_Crysta_2.8_GX_AT_%28facelift%2C_front%29.jpg/320px-Toyota_Innova_Crysta_2.8_GX_AT_%28facelift%2C_front%29.jpg",
    "Toyota Fortuner":   "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/2021_Toyota_Fortuner_Legender_4x2_AT_%28front%29.jpg/320px-2021_Toyota_Fortuner_Legender_4x2_AT_%28front%29.jpg",
    "Toyota Etios":      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Toyota_Etios_Liva_VXD_%28facelift%2C_front%29.jpg/320px-Toyota_Etios_Liva_VXD_%28facelift%2C_front%29.jpg",
    "Toyota Corolla":    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Toyota_Corolla_Altis_1.8_GL_%28facelift%2C_front%29.jpg/320px-Toyota_Corolla_Altis_1.8_GL_%28facelift%2C_front%29.jpg",
    "Toyota Camry":      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2019_Toyota_Camry_Hybrid_%28India%2C_front%29.jpg/320px-2019_Toyota_Camry_Hybrid_%28India%2C_front%29.jpg",
    "Tata Nexon":        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/2023_Tata_Nexon_Creative%2B_Dark_%28front%29.jpg/320px-2023_Tata_Nexon_Creative%2B_Dark_%28front%29.jpg",
    "Tata Tiago":        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/2020_Tata_Tiago_XZ%2B_%28facelift%2C_front%29.jpg/320px-2020_Tata_Tiago_XZ%2B_%28facelift%2C_front%29.jpg",
    "Tata Punch":        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/2022_Tata_Punch_Creative_AMT_%28front%29.jpg/320px-2022_Tata_Punch_Creative_AMT_%28front%29.jpg",
    "Tata Harrier":      "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/2023_Tata_Harrier_Adventure_%28front%29.jpg/320px-2023_Tata_Harrier_Adventure_%28front%29.jpg",
    "Tata Safari":       "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/2023_Tata_Safari_Adventure_Plus_%28front%29.jpg/320px-2023_Tata_Safari_Adventure_Plus_%28front%29.jpg",
    "Kia Seltos":        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/2022_Kia_Seltos_HTX_Plus_D_%28front%29.jpg/320px-2022_Kia_Seltos_HTX_Plus_D_%28front%29.jpg",
    "Kia Sonet":         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/2021_Kia_Sonet_HTX%2B_Diesel_AT_%28front%29.jpg/320px-2021_Kia_Sonet_HTX%2B_Diesel_AT_%28front%29.jpg",
    "Kia Carens":        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/2022_Kia_Carens_Prestige_Plus_1.5_Diesel_DCT_6AT_%28front%29.jpg/320px-2022_Kia_Carens_Prestige_Plus_1.5_Diesel_DCT_6AT_%28front%29.jpg",
    "MG Hector":         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/2022_MG_Hector_Sharp_1.5T_DCT_%28front%29.jpg/320px-2022_MG_Hector_Sharp_1.5T_DCT_%28front%29.jpg",
    "MG ZS EV":          "https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/2022_MG_ZS_EV_Exclusive_%28front%29.jpg/320px-2022_MG_ZS_EV_Exclusive_%28front%29.jpg",
    "Mahindra XUV300":   "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/2019_Mahindra_XUV300_W8_%28O%29_Diesel_%28front%29.jpg/320px-2019_Mahindra_XUV300_W8_%28O%29_Diesel_%28front%29.jpg",
    "Mahindra Scorpio":  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2022_Mahindra_Scorpio_Classic_S11_%28front%29.jpg/320px-2022_Mahindra_Scorpio_Classic_S11_%28front%29.jpg",
    "Mahindra Thar":     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/2021_Mahindra_Thar_LX_Hard_Top_Diesel_AT_%28front%29.jpg/320px-2021_Mahindra_Thar_LX_Hard_Top_Diesel_AT_%28front%29.jpg",
    "Mahindra XUV700":   "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/2021_Mahindra_XUV700_AX7_L_Diesel_AT_AWD_%28front%29.jpg/320px-2021_Mahindra_XUV700_AX7_L_Diesel_AT_AWD_%28front%29.jpg",
    "Renault Kwid":      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/2022_Renault_Kwid_Climber_AMT_%28front%29.jpg/320px-2022_Renault_Kwid_Climber_AMT_%28front%29.jpg",
    "Renault Kiger":     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/2021_Renault_Kiger_RXT_%28O%29_Turbo_CVT_%28front%29.jpg/320px-2021_Renault_Kiger_RXT_%28O%29_Turbo_CVT_%28front%29.jpg",
    "Nissan Magnite":    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/2021_Nissan_Magnite_XV_Premium_%28O%29_Turbo_CVT_%28front%29.jpg/320px-2021_Nissan_Magnite_XV_Premium_%28O%29_Turbo_CVT_%28front%29.jpg",
    "Skoda Kushaq":      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/2021_Skoda_Kushaq_Style_1.5_TSI_DSG_%28front%29.jpg/320px-2021_Skoda_Kushaq_Style_1.5_TSI_DSG_%28front%29.jpg",
    "Volkswagen Taigun": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/2021_Volkswagen_Taigun_GT_Plus_1.5_TSI_DSG_%28front%29.jpg/320px-2021_Volkswagen_Taigun_GT_Plus_1.5_TSI_DSG_%28front%29.jpg",
}

DEFAULT_CAR_IMG = "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400&q=80"

def get_car_image(car_name: str) -> str:
    if car_name in CAR_IMAGES:
        return CAR_IMAGES[car_name]
    name_lower = car_name.lower()
    for key, url in CAR_IMAGES.items():
        parts = key.lower().split()
        if len(parts) > 1 and parts[1] in name_lower:
            return url
    return DEFAULT_CAR_IMG

# ── THEME TOKENS ──────────────────────────────────────────────────────────────
accent    = "#C0392B"
accent_lt = "#FDECEA"
muted     = "#6B7280"
body      = "#111827"
card_bg   = "#F9FAFB"
border_c  = "#E5E7EB"

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
.block-container {{ padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1280px; }}

/* HEADER */
.site-title {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 58px; font-weight: 900;
    color: {accent}; letter-spacing: -1px;
    margin: 0; padding: 0; line-height: 1;
    text-transform: uppercase;
}}
.site-subtitle {{
    color: {muted}; font-size: 15px; font-weight: 500;
    margin-top: 6px; letter-spacing: 0.3px;
}}
.nav-chip {{
    display: inline-block; background: {accent_lt};
    color: {accent}; font-size: 11px; font-weight: 800;
    letter-spacing: 1px; text-transform: uppercase;
    border-radius: 999px; padding: 5px 14px;
    border: 1px solid rgba(192,57,43,0.2); margin-left: 6px;
}}
.divider {{
    height: 3px;
    background: linear-gradient(90deg, {accent} 0%, {accent_lt} 55%, transparent 100%);
    border: none; margin: 1rem 0 1.5rem; border-radius: 2px;
}}

/* SECTION LABEL */
.section-label {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 13px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 2px;
    color: {muted}; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}}
.section-label::before {{
    content: ''; display: inline-block;
    width: 18px; height: 3px;
    background: {accent}; border-radius: 2px; flex-shrink: 0;
}}

/* METRIC CARDS */
.metric-card {{
    background: {card_bg}; border-radius: 16px;
    padding: 18px 20px; border: 1.5px solid {border_c};
    text-align: center;
}}
.metric-card.hi {{ border-color: {accent}; background: {accent_lt}; }}
.mc-model {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: {muted}; }}
.mc-label {{ font-size: 11px; color: {muted}; margin-top: 2px; }}
.mc-val {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 32px; font-weight: 900; color: {accent};
    margin-top: 4px; line-height: 1; letter-spacing: -0.5px;
}}

/* RESULT CARDS */
.result-card {{
    background: {card_bg}; border-radius: 16px;
    padding: 22px; border: 1.5px solid {border_c}; margin-bottom: 14px;
}}
.result-card.best {{ border-color: {accent}; background: {accent_lt}; }}
.rc-model {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: {muted}; margin-bottom: 4px; }}
.rc-price {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 40px; font-weight: 900; color: {accent};
    line-height: 1; letter-spacing: -1px;
}}
.rc-pill {{
    display: inline-block; background: {accent}; color: #fff;
    font-size: 10px; font-weight: 800; letter-spacing: 1.2px;
    text-transform: uppercase; border-radius: 999px;
    padding: 4px 12px; margin-top: 10px;
}}
.empty-state {{
    background: {card_bg}; border-radius: 16px;
    padding: 48px 24px; border: 1.5px dashed {border_c}; text-align: center;
}}
.empty-icon {{ font-size: 40px; margin-bottom: 12px; }}
.empty-text {{ font-size: 14px; color: {muted}; font-weight: 500; }}
.empty-cta {{ color: {accent}; font-weight: 700; }}

/* CAR CARDS */
.car-card {{
    background: #fff; border-radius: 20px;
    border: 1.5px solid {border_c}; overflow: hidden; height: 100%;
}}
.car-img-wrap {{
    width: 100%; height: 175px;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    overflow: hidden; position: relative;
}}
.car-img-wrap img {{
    width: 100%; height: 100%; object-fit: cover; object-position: center;
}}
.car-fuel-badge {{
    position: absolute; top: 10px; left: 10px;
    background: {accent}; color: #fff;
    font-size: 10px; font-weight: 800; letter-spacing: 1px;
    text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
}}
.car-body {{ padding: 16px 18px 20px; }}
.car-brand {{
    font-size: 11px; font-weight: 700; color: {muted};
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 2px;
}}
.car-name {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800; font-size: 24px; color: {body};
    line-height: 1.1; letter-spacing: -0.3px; margin-bottom: 8px;
}}
.car-desc {{ font-size: 13px; color: #4B5563; line-height: 1.55; margin-bottom: 12px; }}
.badge {{
    display: inline-block; background: {accent_lt};
    border: 1px solid rgba(192,57,43,0.2); color: {accent};
    border-radius: 999px; padding: 3px 10px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 4px; margin-top: 4px;
}}
.no-result {{
    text-align: center; padding: 48px 24px;
    border: 1.5px dashed {border_c}; border-radius: 16px;
    color: {muted}; font-size: 14px; font-weight: 500;
}}

/* NATIVE STREAMLIT OVERRIDES */
label, .stSlider label {{
    color: {muted} !important; font-size: 12px !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background: #fff !important; color: {body} !important;
    border: 1.5px solid {border_c} !important;
    border-radius: 10px !important; font-size: 15px !important; font-weight: 600 !important;
}}
.stSelectbox > div > div {{
    background: #fff !important; border: 1.5px solid {border_c} !important;
    border-radius: 10px !important; color: {body} !important;
    font-size: 15px !important; font-weight: 600 !important;
}}
.stSlider > div > div > div > div {{ background: {accent} !important; }}

/* TABS */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px; background: #F3F4F6; border-radius: 12px;
    padding: 5px; border: 1.5px solid {border_c};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {muted}; border-radius: 8px;
    font-size: 14px; font-weight: 700; padding: 10px 26px;
}}
.stTabs [aria-selected="true"] {{
    background: {accent} !important; color: #fff !important;
}}

/* BUTTON — fixed cut-off issue */
div[data-testid="stFormSubmitButton"] {{ width: 100%; }}
div[data-testid="stFormSubmitButton"] > button {{
    background: {accent} !important; color: #fff !important;
    width: 100% !important; padding: 15px 24px !important;
    border-radius: 12px !important; border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900 !important; font-size: 18px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(192,57,43,0.28) !important;
    transition: opacity 0.2s !important; display: block !important;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{ opacity: 0.88 !important; }}

.stAlert {{ border-radius: 12px !important; font-weight: 500 !important; }}
.footer {{
    text-align: center; color: {muted}; font-size: 12px;
    font-weight: 500; padding: 16px 0 4px; letter-spacing: 0.3px;
}}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("<div class='site-title'>SmartCar AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='site-subtitle'>Used Car Price Prediction &nbsp;·&nbsp; ML-Powered Insights</div>", unsafe_allow_html=True)
with h2:
    st.markdown("""<div style='text-align:right; margin-top:20px;'>
        <span class='nav-chip'>CarDekho</span>
        <span class='nav-chip'>2 Models</span>
    </div>""", unsafe_allow_html=True)

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
            better  = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"

            st.markdown(f"""
<div class='result-card'>
  <div class='rc-model'>Linear Regression</div>
  <div class='rc-price'>₹ {lr_pred:.2f} <span style='font-size:17px;font-weight:600;'>Lakh</span></div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""
<div class='result-card best'>
  <div class='rc-model'>Random Forest</div>
  <div class='rc-price'>₹ {rf_pred:.2f} <span style='font-size:17px;font-weight:600;'>Lakh</span></div>
  <div class='rc-pill'>✦ Most Accurate</div>
</div>""", unsafe_allow_html=True)

            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by ₹{diff:.2f}L · Best R² = {max(lr_r2, rf_r2):.3f}")
        else:
            st.markdown("""
<div class='empty-state'>
  <div class='empty-icon'>🔮</div>
  <div class='empty-text'>Fill in car details and hit<br>
    <span class='empty-cta'>Get Prediction</span>
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — CAR RECOMMENDER
# ═══════════════════════════════════════════════════════════════
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
                format_func=lambda x: {"any": "No Preference", "petrol": "Petrol",
                                        "diesel": "Diesel", "electric": "Electric"}[x])
        with c2:
            usage = st.selectbox("Usage", ["city", "highway", "both"],
                format_func=lambda x: {"city": "Mostly City", "highway": "Mostly Highway", "both": "Both"}[x])
            family_size = st.selectbox("Family Size", ["1-2", "3-4", "5+"],
                format_func=lambda x: {"1-2": "1–2 People", "3-4": "3–4 People", "5+": "5+ People"}[x])
        with c3:
            priority = st.selectbox("Priority", ["mileage", "comfort", "style", "performance"],
                format_func=lambda x: {"mileage": "Best Mileage", "comfort": "Comfort & Space",
                                        "style": "Stylish Looks", "performance": "Performance"}[x])
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
            st.markdown("<div class='no-result'>No exact match found. Try <strong>No Preference</strong> for fuel or adjust your budget.</div>",
                unsafe_allow_html=True)
        else:
            cols = st.columns(len(results))
            for col, (_, row) in zip(cols, results.iterrows()):
                img_url = get_car_image(row['car_name'])
                with col:
                    st.markdown(f"""
<div class='car-card'>
  <div class='car-img-wrap'>
    <img src='{img_url}' alt='{row["car_name"]}'
         onerror="this.style.display='none'"/>
    <div class='car-fuel-badge'>{row['fuel'].upper()}</div>
  </div>
  <div class='car-body'>
    <div class='car-brand'>{row['brand']}</div>
    <div class='car-name'>{row['car_name']}</div>
    <div class='car-desc'>{row['description']}</div>
    <div>
      <span class='badge'>{row['budget'].replace('_',' ')}</span>
      <span class='badge'>{row['usage'].capitalize()}</span>
      <span class='badge'>{row['priority'].capitalize()}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>SmartCar AI &nbsp;·&nbsp; CarDekho Dataset &nbsp;·&nbsp; Linear Regression + Random Forest</div>",
    unsafe_allow_html=True)
