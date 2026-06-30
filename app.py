import streamlit as st
import joblib
from src.data_loader import load_datasets
from src.features import build_features

st.set_page_config(layout="wide")

model = joblib.load("rf_model.pkl")
features = joblib.load("features.pkl")

equipos, fixture = load_datasets()

st.title("🏆 Mundial 2026 Simulator")

grupo = st.selectbox("Grupo", sorted(equipos["group"].unique()))

st.write("Equipos:")
st.dataframe(equipos[equipos["group"] == grupo])

if st.button("Simular"):
    st.success("Simulación lista")

    partidos = list(zip(fixture["home"], fixture["away"]))

    for a, b in partidos[:10]:
        X = build_features(a, b, equipos)
        probs = model.predict_proba(X)[0]

        st.write(a, "vs", b)
        st.write(probs)