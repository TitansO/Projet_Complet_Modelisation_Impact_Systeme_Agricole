"""
Climat IA — Interface Streamlit
Dashboard de visualisation des profils de vulnérabilité agronomique
pour les 3 systèmes de production agricole (BAU, Vegan/Polyculture, ICLS).
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://reverse_proxy/api")

st.set_page_config(
    page_title="Climat IA — Vulnérabilité Agronomique",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Sidebar — Paramètres de prédiction
# ─────────────────────────────────────────────
st.sidebar.title("🌾 Climat IA")
st.sidebar.markdown("**Modélisation de l'impact climatique sur les rendements agricoles**")
st.sidebar.markdown("---")

systeme = st.sidebar.selectbox(
    "Système de production",
    ["Systeme_Grandes_Cultures", "Systeme_Polyculture_Elevage", "Systeme_Agroecologique"],
    help="Correspond aux systèmes BAU, Vegan/ICLS du TFE Delandmeter (2021)",
)

scenario_rcp = st.sidebar.selectbox(
    "Scénario RCP",
    ["RCP_4.5", "RCP_8.5"],
    help="RCP 4.5 = scénario modéré | RCP 8.5 = scénario pessimiste (business-as-usual global)",
)

st.sidebar.markdown("### Variables climatiques")
temperature = st.sidebar.slider("Température (°C)", -5.0, 45.0, 22.0, 0.5)
pluviometrie = st.sidebar.slider("Pluviométrie (mm/an)", 100.0, 2000.0, 750.0, 10.0)
indice_secheresse = st.sidebar.slider("Indice de sécheresse", 0.0, 1.0, 0.35, 0.01)
radiation = st.sidebar.slider("Radiation solaire (kWh/m²)", 50.0, 300.0, 130.0, 5.0)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🌍 Climat IA — Tableau de Bord de Vulnérabilité Agronomique")
st.markdown(
    "Architecture microservices polyglotte · PostgreSQL · Apache Cassandra · Redis · FastAPI"
)
st.markdown("---")

# ─────────────────────────────────────────────
# Appel API — Prédiction
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("🔮 Lancer la prédiction de vulnérabilité", type="primary", use_container_width=True):
        payload = {
            "systeme_production": systeme,
            "scenario_rcp": scenario_rcp,
            "temperature_c": temperature,
            "pluviometrie_mm": pluviometrie,
            "indice_secheresse": indice_secheresse,
            "radiation_kwh": radiation,
        }
        try:
            with st.spinner("Interrogation de l'API (vérification cache Redis → PostgreSQL)…"):
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.session_state["prediction"] = data
                st.success("✅ Prédiction reçue")
            else:
                st.error(f"Erreur API {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ API indisponible — vérifier que la stack Docker est lancée.")

with col2:
    if st.button("🔄 Santé de l'API", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                health = r.json()
                st.success(f"API OK · DB: {health.get('postgres')} · Cache: {health.get('redis')}")
            else:
                st.error("API non disponible")
        except Exception:
            st.warning("Connexion impossible")

# ─────────────────────────────────────────────
# Résultats de prédiction
# ─────────────────────────────────────────────
if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    st.markdown("### 📊 Résultats de la Prédiction")

    cols = st.columns(4)
    cols[0].metric("Cluster K-Means", f"#{pred.get('cluster_id', '?')}")
    cols[1].metric("Profil climatique", pred.get("profil_climatique", "—"))
    cols[2].metric("Impact rendement", pred.get("impact_rendement", "—"))
    cols[3].metric(
        "Cache",
        "🟡 HIT Redis" if pred.get("cache_hit") else "🔵 MISS → PostgreSQL",
    )

    st.info(
        f"**Système :** {pred.get('systeme_production')} | "
        f"**Scénario :** {pred.get('scenario_rcp')} | "
        f"**Latence :** {pred.get('latence_ms', '—')} ms"
    )

# ─────────────────────────────────────────────
# Radar chart — Profil des features climatiques
# ─────────────────────────────────────────────
st.markdown("### 🕸️ Profil Climatique Courant")
features = {
    "Température": temperature / 45,
    "Pluviométrie": pluviometrie / 2000,
    "Sécheresse": indice_secheresse,
    "Radiation": radiation / 300,
}
fig_radar = go.Figure(
    data=go.Scatterpolar(
        r=list(features.values()),
        theta=list(features.keys()),
        fill="toself",
        line_color="#2ECC71",
    )
)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    height=350,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# Comparaison des 3 systèmes agricoles (TFE)
# ─────────────────────────────────────────────
st.markdown("### 🌱 Comparaison des 3 Systèmes Agricoles (TFE Delandmeter, 2021)")
df_systemes = pd.DataFrame(
    {
        "Système": [
            "Grandes Cultures (BAU)",
            "Polyculture-Élevage (Vegan)",
            "Agroécologique (ICLS)",
        ],
        "Rendement relatif": [1.00, 0.88, 0.85],
        "Résilience hydrique": [0.45, 0.65, 0.92],
        "Stock carbone sol": [0.58, 0.30, 0.95],
        "Auto-suffisance": [0.50, 0.70, 0.90],
    }
)
fig_bar = px.bar(
    df_systemes.melt(id_vars="Système"),
    x="variable",
    y="value",
    color="Système",
    barmode="group",
    labels={"variable": "Indicateur", "value": "Score normalisé"},
    color_discrete_sequence=["#3498DB", "#E67E22", "#2ECC71"],
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption(
    "Climat IA · Architecture Polyglotte (PostgreSQL + Cassandra + Redis) · "
    "ESMT Dakar 2025 · Données issues du TFE Delandmeter (2021), ULiège GxABT"
)
