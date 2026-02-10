import streamlit as st
import fastf1
from fastf1_utils.fastf1_service import get_race_summary

st.set_page_config(page_title="F1 Prediction App", layout="wide")
st.title("F1 Prediction Game")

st.sidebar.header("Race Selection")
season = st.sidebar.selectbox("Season", list(range(2018, 2025)))


@st.cache_data(show_spinner=False)
def load_event_names(year: int) -> list[str]:
    schedule = fastf1.get_event_schedule(year)
    return schedule["EventName"].dropna().tolist()


event_names = load_event_names(season)
if event_names:
    gp_name = st.sidebar.selectbox("Grand Prix", event_names, index=0)
else:
    gp_name = None
    st.sidebar.warning("No events found for this season.")

if st.sidebar.button("Load race summary", disabled=gp_name is None):
    with st.spinner("Loading race data..."):
        results, weather = get_race_summary(season, gp_name)

    st.subheader(f"{gp_name} {season} (Top 10)")
    st.dataframe(results, use_container_width=True)

    st.subheader("Weather summary")
    st.write(weather)
else:
    st.info("Select a season and GP name, then click 'Load race summary'.")
