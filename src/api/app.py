# app.py
import streamlit as st

st.set_page_config(
    page_title="F1 Prediction App",
    layout="wide"
)

st.title("F1 Prediction Game using Machine Learning")
st.write(
    "Welcome! Follow the steps in the sidebar to pick a race, adjust the "
    "parameters you care about, and see how our model predicts the finishing "
    "order.\n\nReady when you are!"
)
