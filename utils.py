import os
import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

DATA_DIR = "data"
MODEL_DIR = "models"

EPC_MAP = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}

EPC_COLORS = {
    "A": "#009A44", "B": "#4CB748", "C": "#AACC29",
    "D": "#FFB612", "E": "#FCAA65", "F": "#EF8023", "G": "#ED1C24",
}


def load_css():
    with open(os.path.join("assets", "style.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-brand">Green Premium</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-tagline">Greater Manchester house price analytics</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption("CS648 Project · Bharath Balur Umashankar")


def epc_strip_html(active_letter):
    segments = ""
    for letter in ["A", "B", "C", "D", "E", "F", "G"]:
        cls = "epc-segment active" if letter == active_letter else "epc-segment"
        segments += f'<div class="{cls}" style="background:{EPC_COLORS[letter]}">{letter}</div>'
    return f'<div class="epc-strip">{segments}</div>'


@st.cache_resource
def load_models():
    xgb_model = XGBRegressor()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model_adjusted.json"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_adjusted.pkl"))
    ridge_model = joblib.load(os.path.join(MODEL_DIR, "ridge_adjusted.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_adjusted.pkl"))
    model_features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    return xgb_model, rf_model, ridge_model, scaler, model_features


@st.cache_data
def load_sector_lookup():
    return pd.read_parquet(os.path.join(DATA_DIR, "phase6_sector_lookup.parquet"))


@st.cache_data
def load_adj_factor_table():
    return pd.read_parquet(os.path.join(DATA_DIR, "phase6_adj_factor_table.parquet"))


@st.cache_data
def load_sector_map_data():
    return pd.read_parquet(os.path.join(DATA_DIR, "phase6_sector_map_data.parquet"))


@st.cache_data
def load_model_metrics():
    return pd.read_parquet(os.path.join(DATA_DIR, "phase6_model_metrics.parquet"))


@st.cache_data
def load_feature_importance():
    return pd.read_parquet(os.path.join(DATA_DIR, "phase6_feature_importance.parquet"))


@st.cache_data
def load_sector_avg_price():
    return pd.read_parquet(os.path.join(DATA_DIR, "phase6_sector_avg_price.parquet"))


@st.cache_resource
def load_gm_median_price():
    return joblib.load(os.path.join(DATA_DIR, "phase6_gm_median_price.pkl"))


def get_adj_factor(district, year_month, adj_table):
    subset = adj_table[adj_table["postcode_district"] == district].copy()
    if subset.empty:
        return 1.0
    subset["distance"] = (subset["year_month"] - year_month).abs()
    return subset.loc[subset["distance"].idxmin(), "adj_factor"]