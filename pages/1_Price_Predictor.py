from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    EPC_MAP,
    epc_strip_html,
    get_adj_factor,
    load_adj_factor_table,
    load_css,
    load_gm_median_price,
    load_models,
    load_sector_avg_price,
    load_sector_lookup,
    render_sidebar,
)

st.set_page_config(page_title="Price Predictor", layout="wide")
load_css()
render_sidebar()

xgb_model, rf_model, ridge_model, scaler, model_features = load_models()
sector_lookup = load_sector_lookup()
adj_factor_table = load_adj_factor_table()
sector_avg_price = load_sector_avg_price()
gm_median_price = load_gm_median_price()

ENSEMBLE_MAPE = 18.73

PROPERTY_TYPES = {
    "Detached": "prop_type_D", "Semi-detached": "prop_type_S",
    "Terraced": "prop_type_T", "Other (bungalow, etc.)": "prop_type_O",
    "Flat / Maisonette": None,
}
AGE_BANDS = {
    "Before 1919": "age_pre_1919", "1919–1944": "age_yr1919_1944",
    "1945–1972": "age_yr1945_1972", "1973–1990": "age_yr1973_1990",
    "1991–2010": "age_yr1991_2010", "After 2010": None,
}


def build_features(epc_letter_value, sector_row, floor_area, prop_type, age_band,
                    tenure, is_new_build, sale_date):
    row = {
        "epc_numeric": EPC_MAP[epc_letter_value], "total_floor_area": floor_area,
        "dist_city_centre_km": sector_row["dist_city_centre_km"],
        "dist_station_km": sector_row["dist_station_km"],
        "spatial_lag_price": sector_row["spatial_lag_price"],
        "sale_year": sale_date.year, "sale_month": sale_date.month,
        "is_new_build": int(is_new_build),
        "tenure_encoded": 1 if tenure == "Freehold" else 0,
        "prop_type_D": 0, "prop_type_S": 0, "prop_type_T": 0, "prop_type_O": 0,
        "age_pre_1919": 0, "age_yr1919_1944": 0, "age_yr1945_1972": 0,
        "age_yr1973_1990": 0, "age_yr1991_2010": 0,
    }
    prop_col = PROPERTY_TYPES[prop_type]
    if prop_col:
        row[prop_col] = 1
    age_col = AGE_BANDS[age_band]
    if age_col:
        row[age_col] = 1
    return pd.DataFrame([row])[model_features]


def predict_price(epc_letter_value, adj_factor, **kwargs):
    X = build_features(epc_letter_value, **kwargs)
    xgb_p = xgb_model.predict(X)[0]
    rf_p = rf_model.predict(X)[0]
    return np.exp((xgb_p + rf_p) / 2) / adj_factor


st.markdown('<div class="hero-subtitle">Live Model · Ensemble (XGBoost + Random Forest)</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Property Price Estimator</div>', unsafe_allow_html=True)

if "result" in st.session_state:
    r = st.session_state["result"]

    st.markdown(
        f'<div class="result-hero">'
        f'<div class="stat-label">Estimated Market Value</div>'
        f'<div class="value-display">£{r["price"]:,.0f}</div>'
        f'<div class="value-caption">Typical prediction error: ±{ENSEMBLE_MAPE:.2f}% &nbsp;·&nbsp; '
        f'Illustrative range: £{r["low"]:,.0f} – £{r["high"]:,.0f}</div>'
        f'{epc_strip_html(r["epc_letter"])}'
        f'</div>', unsafe_allow_html=True
    )

    if not r["standard_sale"]:
        st.warning(
            "Predictions for non-standard sales are less reliable in our testing: "
            "28.68% average error, versus 16.59% for standard open-market sales."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">How this compares</div>', unsafe_allow_html=True)
        comparison = pd.DataFrame({
            "Value": [r["price"], r["sector_median"], r["gm_median"]]
        }, index=["This prediction", f"{r['sector']} historical median", "Greater Manchester median"])
        st.bar_chart(comparison)
    with c2:
        st.markdown('<div class="section-label">Green Premium by EPC rating</div>', unsafe_allow_html=True)
        epc_df = pd.DataFrame({"Estimated price": r["prices_by_letter"]}).rename_axis("EPC rating")
        st.bar_chart(epc_df)

    rows_html = ""
    letters = list(r["prices_by_letter"].keys())
    for prev_letter, next_letter in zip(letters, letters[1:]):
        delta = r["prices_by_letter"][next_letter] - r["prices_by_letter"][prev_letter]
        rows_html += (
            f'<div class="delta-row"><span>{prev_letter} &rarr; {next_letter}</span>'
            f'<span class="delta-value">+£{delta:,.0f}</span></div>'
        )
    st.markdown('<div class="section-label">EPC upgrade breakdown</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

form_expanded = "result" not in st.session_state
with st.expander("Enter property details", expanded=form_expanded):
    show_advanced = st.toggle("Show advanced options")

    with st.form("predictor_form"):
        c1, c2 = st.columns(2)
        with c1:
            sector = st.selectbox("Postcode sector", sorted(sector_lookup["postcode_sector"]))
            floor_area = st.number_input("Total floor area (sqm)", min_value=20, max_value=500, value=80)
        with c2:
            epc_letter = st.select_slider("EPC rating", options=["G", "F", "E", "D", "C", "B", "A"], value="D")
            prop_type = st.selectbox("Property type", list(PROPERTY_TYPES.keys()))

        if show_advanced:
            st.markdown('<div class="section-label">Advanced options</div>', unsafe_allow_html=True)
            age_band = st.selectbox("Construction age band", list(AGE_BANDS.keys()))
            tenure = st.radio("Tenure", ["Freehold", "Leasehold"])
            is_new_build = st.checkbox("New build")
            sale_date = st.date_input("Expected sale date", value=date.today(),
                                       min_value=date(2018, 1, 1), max_value=date(2026, 12, 31))
            standard_sale = st.checkbox("This is expected to be a standard open-market sale", value=True)
            st.caption("Repossessions, transfers between related parties, and portfolio sales are "
                       "examples of non-standard transactions.")
        else:
            age_band = "After 2010"
            tenure = "Freehold"
            is_new_build = False
            sale_date = date.today()
            standard_sale = True

        submitted = st.form_submit_button("Estimate Market Value", use_container_width=True)

if submitted:
    sector_row = sector_lookup[sector_lookup["postcode_sector"] == sector].iloc[0]
    year_month = sale_date.year * 12 + sale_date.month
    adj_factor = get_adj_factor(sector_row["postcode_district"], year_month, adj_factor_table)

    kwargs = dict(sector_row=sector_row, floor_area=floor_area, prop_type=prop_type,
                  age_band=age_band, tenure=tenure, is_new_build=is_new_build, sale_date=sale_date)

    predicted_price = predict_price(epc_letter, adj_factor, **kwargs)
    letters = ["G", "F", "E", "D", "C", "B", "A"]
    prices_by_letter = {letter: predict_price(letter, adj_factor, **kwargs) for letter in letters}

    sector_price_row = sector_avg_price[sector_avg_price["postcode_sector"] == sector]
    sector_median = (sector_price_row["sector_median_price"].iloc[0] / adj_factor
                      if not sector_price_row.empty else np.nan)

    st.session_state["result"] = {
        "price": predicted_price,
        "low": predicted_price * (1 - ENSEMBLE_MAPE / 100),
        "high": predicted_price * (1 + ENSEMBLE_MAPE / 100),
        "epc_letter": epc_letter,
        "standard_sale": standard_sale,
        "sector": sector,
        "sector_median": sector_median,
        "gm_median": gm_median_price / adj_factor,
        "prices_by_letter": prices_by_letter,
    }
    st.rerun()
