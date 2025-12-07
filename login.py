from __future__ import annotations
import streamlit as st
from core.security.auth import login_form

st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")

login_form()
