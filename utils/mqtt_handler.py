# utils/mqtt_handler.py

from paho.mqtt import client as mqtt_client
import json

broker = 'localhost'
port = 1883
topic = "lab/transfo/bench1/tr_230_20/measures"
latest_data = {}

def on_message(client, userdata, msg):
    global latest_data
    try:0
        payload = json.loads(msg.payload.decode())
        latest_data = payload
    except Exception:
        pass

def connect_mqtt():
    client = mqtt_client.Client("streamlit-client")
    client.on_message = on_message
    try:
        client.connect(broker, port)
        client.subscribe(topic)
        client.loop_start()
        return True
    except Exception as e:
        print("MQTT connection failed:", e)
        return False

def start_mqtt_loop():
    pass  # déjà lancé dans connect_mqtt

def get_latest_measure():
    return latest_data
