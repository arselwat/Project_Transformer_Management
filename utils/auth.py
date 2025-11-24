import streamlit as st

def check_password() -> bool:
    """
    Affiche une jolie fenêtre de login et renvoie True si l'utilisateur est connecté.
    À appeler au début de chaque page.
    """

    # Déjà loggé ?
    if st.session_state.get("password_correct", False):
        return True

    # Callback appelé quand on soumet le formulaire
    def password_entered():
        user = st.session_state.get("username", "")
        pwd = st.session_state.get("password", "")

        # Récupérer les identifiants depuis les secrets
        valid_user = st.secrets["auth"].get("user", "")
        valid_pwd = st.secrets["auth"].get("password", "")

        if user == valid_user and pwd == valid_pwd:
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = user
        else:
            st.session_state["password_correct"] = False

    # --- UI jolie du login ---
    st.markdown(
        """
        <style>
        .login-card {
            max-width: 420px;
            margin: 60px auto;
            padding: 30px 30px 25px 30px;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 10px 25px rgba(15, 23, 42, .18);
        }
        .login-title {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .login-subtitle {
            text-align: center;
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown(
            "<div class='login-title'>🔐 Connexion</div>"
            "<div class='login-subtitle'>Accès au tableau de bord de fiabilité</div>",
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            st.text_input("Nom d'utilisateur", key="username")
            st.text_input("Mot de passe", type="password", key="password")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Se connecter", use_container_width=True, on_click=password_entered)

        # Message d'erreur si mauvais logins
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("Identifiants incorrects. Réessaie.")

        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.get("password_correct", False)
