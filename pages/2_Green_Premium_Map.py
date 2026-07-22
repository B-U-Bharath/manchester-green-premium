import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils import load_css, load_sector_map_data, render_sidebar

st.set_page_config(page_title="Green Premium Map", layout="wide")
load_css()
render_sidebar()

st.markdown('<div class="hero-subtitle">Sector-Level Analysis · 354 Postcode Sectors</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Green Premium Map</div>', unsafe_allow_html=True)
st.markdown(
    "Each marker shows the estimated effect of upgrading a property from EPC rating D to B, "
    "holding location, size, and property type fixed. Darker green means a stronger positive "
    "effect in that area; red means the effect is negative there."
)

sector_map_data = load_sector_map_data()

last_sector = None
if "result" in st.session_state:
    last_sector = st.session_state["result"]["sector"]
    st.info(f"Centred on **{last_sector}** — your most recent Price Predictor search.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Highest premium</div>'
        f'<div class="stat-number">{sector_map_data["green_premium_pct"].max():.2f}%</div></div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Lowest premium</div>'
        f'<div class="stat-number">{sector_map_data["green_premium_pct"].min():.2f}%</div></div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Sectors mapped</div>'
        f'<div class="stat-number">{len(sector_map_data)}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


def premium_color(pct, vmin, vmax):
    if pct >= 0:
        intensity = pct / vmax if vmax > 0 else 0
        green_val = int(80 + intensity * 100)
        return f"#{max(0,120-green_val)//2:02x}{min(154,green_val+60):02x}{60:02x}"
    else:
        intensity = abs(pct) / abs(vmin) if vmin < 0 else 0
        red_val = int(140 + intensity * 100)
        return f"#{min(220,red_val):02x}{60:02x}{60:02x}"


vmin = sector_map_data["green_premium_pct"].min()
vmax = sector_map_data["green_premium_pct"].max()

if last_sector is not None and last_sector in sector_map_data["postcode_sector"].values:
    center_row = sector_map_data[sector_map_data["postcode_sector"] == last_sector].iloc[0]
    map_center = [center_row["latitude"], center_row["longitude"]]
    zoom = 13
else:
    map_center = [53.4808, -2.2426]
    zoom = 10

m = folium.Map(location=map_center, zoom_start=zoom, tiles="CartoDB positron")

for _, row in sector_map_data.iterrows():
    is_selected = row["postcode_sector"] == last_sector
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=10 if is_selected else 6,
        color="#14231C" if is_selected else premium_color(row["green_premium_pct"], vmin, vmax),
        fill=True,
        fill_color=premium_color(row["green_premium_pct"], vmin, vmax),
        fill_opacity=1.0 if is_selected else 0.85,
        weight=3 if is_selected else 1,
        popup=folium.Popup(
            f"<b>{row['postcode_sector']}</b><br>"
            f"Green Premium: {row['green_premium_pct']:.2f}%<br>"
            f"Properties: {int(row['n_properties'])}",
            max_width=200,
        ),
    ).add_to(m)

st_folium(m, use_container_width=True, height=560)

st.markdown('<div class="section-label">Top and bottom sectors</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Highest Green Premium**")
    top = sector_map_data.sort_values("green_premium_pct", ascending=False).head(5)
    st.dataframe(top[["postcode_sector", "green_premium_pct", "n_properties"]], hide_index=True, use_container_width=True)
with c2:
    st.markdown("**Lowest Green Premium**")
    bottom = sector_map_data.sort_values("green_premium_pct").head(5)
    st.dataframe(bottom[["postcode_sector", "green_premium_pct", "n_properties"]], hide_index=True, use_container_width=True)