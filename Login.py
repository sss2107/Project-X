import streamlit as st
from loguru import logger
import sys
from streamlit_extras.switch_page_button import switch_page
import hashlib
import pandas as pd
from st_pages import hide_pages, show_pages, Page, Section

logger.remove()
logger.add(sys.__stdout__)
logger.add("prompt.log", rotation="500 MB")


def hash_w2k(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode("utf-8"))
    hashed_output = md5_hash.hexdigest()
    return hashed_output


# getw2k = GetW2K()

st.set_page_config(page_title="Ask anything from Sahil's resume", page_icon="👋")



# st.set_page_config(page_title=" Aircraft Log Spell Checker", page_icon="😇")
html_temp = """
<div style="background-color:brown;padding:10px">
<h2 style="color:white;text-align:center;"> Ask anything from Sahil's resume👋 </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

# st.write("# Welcome to GPT Studio! 👋")

st.warning("This is an experimental GenAI app restricted for use only for SAP AI lab team. Please do not share this link with anyone. ")


with st.form(key="authentication"):
    usr_input = st.text_input(label="Please input your username")
    #pwd_input = st.text_input(label="Please input your w2k password", type="password")
    department = st.text_input(label="Your Department")
    st.session_state.usr_input=usr_input
    st.session_state.department=department
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    # w2k_check = getw2k.authenticate(user_name=usr_input, pwd=pwd_input)
    if usr_input == "" or department == "":
        st.write("You have input wrong/incomplete info")
        st.session_state.authenticated = False
    else:
        # st.write(f"Welcome {usr_input}")
        st.session_state.authenticated = True
        st.session_state.w2k = usr_input
        # st.session_state.prompt = ""
        st.session_state.w2k_hash = hash_w2k(usr_input)
        st.session_state.dept = department
        logger.info(f"w2k:{usr_input}|")
        switch_page("QA Engine")
