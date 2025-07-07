# lector_compartido.py
from lector_serial import iniciar_puerto, leer_datos
from datetime import datetime
import threading
import time

# Variables globales
datos_actuales = {"timestamp": None, "raw": None, "pct": None}
_lock = threading.Lock()
_running = False

def iniciar_lector():
    global _running
    _running = True
    hilo = threading.Thread(target=_loop_lectura, daemon=True)
    hilo.start()

def detener_lector():
    global _running
    _running = False

def obtener_datos():
    with _lock:
        return datos_actuales.copy()

def _loop_lectura():
    from guardador import init_csv, append_csv  # Importa aquí para evitar problemas circulares
    from graficador import Graficador

    ser = iniciar_puerto()
    if not ser:
        print("❌ No se pudo abrir el puerto serial.")
        return

    init_csv()
    graf = Graficador()

    print("📡 Iniciando lectura compartida del sensor…")

    try:
        while _running:
            datos = leer_datos(ser)
            if datos:
                raw, pct = datos
                ts = datetime.now().strftime("%H:%M:%S")
                with _lock:
                    datos_actuales["timestamp"] = ts
                    datos_actuales["raw"] = raw
                    datos_actuales["pct"] = pct
                print(f"{ts} → raw:{raw}  pct:{pct}%")
                append_csv(ts, raw, pct)
                graf.actualizar(ts, pct)
            time.sleep(0.5)
    except Exception as e:
        print("💥 Error en lectura:", e)
    finally:
        ser.close()
        print("🔌 Lector detenido.")
