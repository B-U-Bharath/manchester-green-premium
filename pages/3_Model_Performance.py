import streamlit as st

from utils import load_css, load_feature_importance, load_model_metrics, render_sidebar

st.set_page_config(page_title="Model Performance", layout="wide")
load_css()
render_sidebar()

st.markdown('<div class="hero-subtitle">Evaluation &amp; Methodology</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown(
    "Four models were trained and evaluated on the same time-adjusted target and the same "
    "held-out test set (72,337 real sales), so the comparison below is fair and consistent."
)

metrics_df = load_model_metrics()

st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)
display_df = metrics_df.copy()
display_df["mape_pct"] = display_df["mape_pct"].map(lambda x: f"{x:.2f}%")
display_df["r2"] = display_df["r2"].map(lambda x: f"{x:.3f}")
display_df.columns = ["Model", "MAPE", "R-squared"]
st.dataframe(display_df, hide_index=True, use_container_width=True)

st.caption(
    "The ensemble (XGBoost + Random Forest, simple average) gives the lowest error and was "
    "used as the live model behind the Price Predictor. Ridge, a simple linear baseline, was "
    "included to confirm the more flexible models genuinely earn their improvement rather than "
    "just fitting noise."
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">What drives the model\'s predictions</div>', unsafe_allow_html=True)
st.markdown(
    "SHAP (SHapley Additive exPlanations) values show how much each feature pushes a "
    "prediction up or down, averaged across all test properties. Property type, floor area, "
    "and local price levels dominate; EPC rating ranks 14th of 18 features here — a real, "
    "modest effect, precisely why the Green Premium was studied separately with the "
    "dedicated statistical analysis referenced elsewhere in this project."
)
st.image("assets/shap_beeswarm_adjusted.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">EPC rating\'s effect in detail</div>', unsafe_allow_html=True)
st.image("assets/shap_dependence_epc_adjusted.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Where the model is less accurate</div>', unsafe_allow_html=True)
st.markdown(
    "Prediction error is not evenly spread across Greater Manchester. This map shows the "
    "model's residuals (predicted minus actual price) by location, in real sale-date terms."
)
st.image("assets/spatial_residual_map_adjusted.png", use_container_width=True)

with st.expander("Secondary technical detail: split-gain feature importance"):
    st.markdown(
        "This is XGBoost's own internal 'gain' measure, a different and less reliable metric "
        "than SHAP for features like EPC rating, since it tends to favour features used for "
        "clean binary splits (like property type) over subtler, more distributed effects. "
        "Shown here for technical completeness, not as the primary evidence."
    )
    importance_df = load_feature_importance()
    st.dataframe(importance_df.head(10), hide_index=True, use_container_width=True)