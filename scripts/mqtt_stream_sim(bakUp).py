# scripts/mqtt_stream_sim.py
from __future__ import annotations
import json, time, math, random, argparse
from paho.mqtt import client as mqtt
import ssl

def synth(t, prev):
    if prev is None:
        v_sec = 20.0 + random.uniform(-0.2, 0.2)
        i_sec = 1.0 + random.uniform(-0.1, 0.1)
        p_sec = v_sec * i_sec
        v_prim = 230.0 + random.uniform(-1.5, 1.5)
        i_prim = p_sec / max(v_prim, 1e-3)
        pf     = 0.95 + random.uniform(-0.02, 0.02)
        temp   = 35.0
        freq   = 50.0 + random.uniform(-0.05, 0.05)
        status = "OK"
    else:
        v_sec = max(19.0, min(21.0, prev["v_sec"] + random.uniform(-0.15, 0.15)))
        load_factor = 0.8 + 0.4*math.sin(t/12.0) + random.uniform(-0.05,0.05)
        i_sec = max(0.2, min(1.6, 1.0*load_factor + random.uniform(-0.05,0.05)))
        p_sec = v_sec*i_sec
        v_prim = max(226.0, min(234.0, 230.0 + 0.2*(v_sec-20.0) + random.uniform(-1.0, 1.0)))
        i_prim = max(0.02, min(1.5, p_sec/max(v_prim,1e-6)))
        pf     = max(0.80, min(1.0, 0.93 + random.uniform(-0.03,0.03)))
        freq   = max(49.8, min(50.2, 50.0 + random.uniform(-0.05,0.05)))
        target = 35.0 + 15.0*load_factor
        temp   = prev["t_core"] + max(-0.2, min(0.2, target - prev["t_core"])) + random.uniform(-0.05,0.05)
        status = "OK"
        if temp > 60 or i_prim > 1.2 or pf < 0.85:
            status = "WARN"
        if temp > 70 or i_prim > 1.4 or pf < 0.80:
            status = "ALARM"

    return {
        "ts": time.time(),
        "v_sec": float(v_sec),
        "i_sec": float(i_sec),
        "p_sec": float(p_sec),
        "t_core": float(temp),
        "v_prim_rms": float(v_prim),
        "i_prim_rms": float(i_prim),
        "pf_prim": float(pf),
        "freq": float(freq),
        "status": status
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--site", default="bench1")
    ap.add_argument("--equip", default="tr_220_20")
    ap.add_argument("--hz", type=int, default=5)
    ap.add_argument("--tls", action="store_true")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    args = ap.parse_args()

    tbase = f"lab/transfo/{args.site}/{args.equip}"
    cli = mqtt.Client()
    if args.user:
        cli.username_pw_set(args.user, args.password)
    if args.tls:
        cli.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)

    cli.will_set(f"{tbase}/state", json.dumps({"status":"OFFLINE"}), qos=1, retain=True)
    cli.connect(args.host, args.port, keepalive=30)
    cli.loop_start()

    cli.publish(f"{tbase}/state", json.dumps({
        "host": f"pi-{args.equip}",
        "fw": "1.0.0",
        "boot_ts": time.time(),
        "sensors": ["INA219","DS18B20","ADS1115"],
        "rate_hz": args.hz,
        "tz": "Africa/Kinshasa",
        "status": "ONLINE"
    }), qos=1, retain=True)

    prev = None
    dt = 1.0/float(max(args.hz,1))
    try:
        while True:
            m = synth(time.time(), prev)
            prev = m
            cli.publish(f"{tbase}/measures", json.dumps(m), qos=0, retain=False)
            # évènement simple
            if m["status"] == "ALARM":
                cli.publish(f"{tbase}/events", json.dumps({
                    "ts": m["ts"], "level":"ALARM", "code":"TEMP_HIGH" if m["t_core"]>70 else "CURRENT_HIGH",
                    "msg": "Seuil dépassé", "value": m["t_core"]
                }), qos=1, retain=False)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        cli.publish(f"{tbase}/state", json.dumps({"status":"OFFLINE"}), qos=1, retain=True)
        cli.loop_stop()
        cli.disconnect()

if __name__ == "__main__":
    main()
