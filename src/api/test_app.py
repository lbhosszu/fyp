import streamlit as st
from fastf1_utils.fastf1_service import get_race_summary

st.set_page_config(page_title="F1 Prediction App", layout="wide")
st.title("F1 Prediction Game (Prototype)")

st.sidebar.header("Race Selection")
season = st.sidebar.selectbox("Season", [2019, 2020, 2021, 2022, 2023, 2024])
gp_name = st.sidebar.text_input("Grand Prix name", "Abu Dhabi")

if st.sidebar.button("Load race summary"):
    with st.spinner("Loading race data..."):
        results, weather = get_race_summary(season, gp_name)

    st.subheader(f"{gp_name} {season} (Top 10)")
    st.dataframe(results, use_container_width=True)

    st.subheader("Weather summary")
    st.write(weather)
else:
    st.info("Select a season and GP name, then click 'Load race summary'.")
