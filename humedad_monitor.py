import serial
import time
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import csv
import os

# — configuración —
PORT       = 'COM7'
BAUDRATE   = 9600
TIMEOUT    = 1
WET, DRY   = 210, 510
CSV_FILE   = 'humedad.csv'
MAX_POINTS = 60

# — función map+constrain de Arduino —
def map_constrain(x, in_min, in_max, out_min, out_max):
    v = (x - in_min)*(out_max - out_min)/(in_max - in_min) + out_min
    return int(max(min(v, out_max), out_min))

# — inicializa CSV si no existe —
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(['timestamp','raw','humidity_pct'])

# — añade un registro al CSV —
def append_csv(ts, raw, pct):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([ts, raw, pct])

# — lee un entero 0–1023 de COM7 o None si no hay dato válido —
def leer_raw():
    try:
        linea = ser.readline().decode('utf-8', errors='ignore').strip()
    except Exception as e:
        print("⚠️ Error lectura serie:", e)
        return None
    if linea.isdigit():
        return int(linea)
    return None

# — script principal —
if __name__ == "__main__":
    # 1) abre el puerto serie
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
        time.sleep(2)   # espera reset Arduino
    except Exception as e:
        print(f"❌ no pude abrir {PORT}: {e}")
        exit(1)

    # 2) prepara CSV y gráfica
    init_csv()
    xs = deque(maxlen=MAX_POINTS)
    ys = deque(maxlen=MAX_POINTS)
    plt.ion()
    fig, ax = plt.subplots()
    fig.show()

    print("⏳ leyendo RAW desde Arduino (analogRead) por COM7… ctrl+c para salir")
    try:
        while True:
            raw = leer_raw()
            if raw is not None:
                pct = map_constrain(raw, WET, DRY, 100, 0)
                ts  = datetime.now().strftime("%H:%M:%S")
                print(f"{ts} → raw:{raw}  pct:{pct}%")

                append_csv(ts, raw, pct)
                xs.append(ts); ys.append(pct)

                ax.clear()
                ax.plot(xs, ys, marker='o')
                ax.set_ylim(0,100)
                ax.set_title("Humedad en tiempo real")
                ax.set_ylabel("Humedad (%)")
                ax.set_xlabel("Hora")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.pause(0.1)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🔌 detenido por usuario, datos en", CSV_FILE)
    finally:
        ser.close()
