# pages/1_Sources_donnees.py
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
import json
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Sources de données", page_icon="📥", layout="wide")
st.title("📥 Sources de données — Import & Capteurs")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MQTT_DATA_FILE = DATA_DIR / "mqtt_measures_log.csv"

# Onglets : CSV manuel ou capteurs
tab_csv, tab_mqtt = st.tabs(["📂 Import CSV", "📡 Données capteurs (MQTT)"])

# =============== Onglet 1 : Import manuel ===============
with tab_csv:
    st.subheader("📂 Import manuel de fichiers CSV")
    file = st.file_uploader("Sélectionnez un fichier CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Aperçu des données importées :", df.head())
        if st.button("💾 Sauvegarder dans le système"):
            path = DATA_DIR / "imported_data.csv"
            df.to_csv(path, index=False)
            st.success(f"Données sauvegardées dans {path}")

# =============== Onglet 2 : Lecture des capteurs en direct ===============
with tab_mqtt:
    st.subheader("📡 Données capteurs en direct via MQTT")

    col1, col2 = st.columns(2)
    with col1:
        mqtt_broker = st.text_input("MQTT Broker", "localhost")
    with col2:
        topic = st.text_input("Topic à écouter", "lab/transfo/#")

    start_mqtt = st.button("▶️ Lancer la réception en direct")

    if start_mqtt:
        placeholder = st.empty()
        mqtt_data = []

        def on_connect(client, userdata, flags, rc):
            placeholder.info("✅ Connecté au broker MQTT")
            client.subscribe(topic)

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                payload["topic"] = msg.topic
                payload["ts_local"] = datetime.now().isoformat()
                mqtt_data.append(payload)
                # Affichage live
                placeholder.json(payload)
                # Ajout au fichier
                df = pd.DataFrame([payload])
                df.to_csv(MQTT_DATA_FILE, mode='a', index=False, header=not MQTT_DATA_FILE.exists())
            except Exception as e:
                placeholder.warning(f"Erreur : {e}")

        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        try:
            client.connect(mqtt_broker, 1883, 60)
            client.loop_start()
        except Exception as e:
            st.error(f"Connexion échouée : {e}")
