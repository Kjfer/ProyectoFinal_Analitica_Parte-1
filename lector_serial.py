import serial
import time
import re

PORT     = 'COM7'
BAUDRATE = 9600
TIMEOUT  = 1

def iniciar_puerto():
    try:
        s = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
        time.sleep(2)   # deja que Arduino arranque
        return s
    except Exception as e:
        print(f"❌ No se pudo abrir {PORT}: {e}")
        return None

def leer_datos(ser):
    """
    Lee una línea como "Raw: 603  |  H2O%: 45%" y devuelve (raw:int, pct:int).
    Si no coincide el formato o hay error, retorna None.
    """
    if ser is None:
        return None
    linea = ser.readline().decode('utf-8', errors='ignore').strip()
    # extraemos ambos números
    m = re.search(r'Raw:\s*(-?\d+).*H2O%:\s*(-?\d+)%', linea)
    if not m:
        return None
    try:
        raw = int(m.group(1))
        pct = int(m.group(2))
        return raw, pct
    except ValueError:
        return None
