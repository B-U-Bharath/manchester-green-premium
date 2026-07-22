import streamlit as st
from utils import load_css, render_sidebar

st.set_page_config(page_title="Greater Manchester House Prices", layout="wide")
load_css()
render_sidebar()

st.markdown('<div class="hero-subtitle">CS648 Project · Residential Property Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Quantifying the Green Premium</div>', unsafe_allow_html=True)
st.markdown(
    "Built on **361,684 real house sales** across Greater Manchester, combining HM Land "
    "Registry price data, EPC energy ratings, and ONS postcode geography.\n\n"
    "Does a better energy efficiency rating add measurable value to a home, once location, "
    "size, and property type are accounted for? This tool lets you test it directly."
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Real sales analysed</div>'
        '<div class="stat-number">361,684</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Postcode sectors mapped</div>'
        '<div class="stat-number">354</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Model accuracy (MAPE)</div>'
        '<div class="stat-number">18.73%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Explore the tool</div>', unsafe_allow_html=True)

st.page_link("pages/1_Price_Predictor.py", label="**Price Predictor** — get an estimated value with a live EPC comparison")