from __future__ import annotations
import streamlit as st
import hashlib

# À mettre plus tard dans st.secrets ou un fichier config. [web:51]
VALID_USERS = {
    "admin": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
    # hash SHA256("password")
}

def _hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()

def login_form():
    st.title("🔐 Connexion")
    user = st.text_input("Utilisateur")
    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter", type="primary"):
        if user in VALID_USERS and _hash_password(pwd) == VALID_USERS[user]:
            st.session_state["auth_ok"] = True
            st.session_state["user"] = user
            st.success("Connecté. Utilisez le menu pour accéder aux pages.")
            st.experimental_rerun()
        else:
            st.error("Identifiants invalides.")

def require_login():
    if st.session_state.get("auth_ok"):
        st.sidebar.success(f"Connecté : {st.session_state.get('user','?')}")
        if st.sidebar.button("Se déconnecter"):
            for k in ["auth_ok", "user"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()
        return

    st.error("Accès restreint. Veuillez d’abord vous connecter depuis la page de login.")
    st.stop()
