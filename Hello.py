import streamlit as st

# pip list --format=freeze > requirements.txt

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)



st.write("# Welcome to ISOM4030! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Demos for important machine learning and deep learning concepts!
    """
)


