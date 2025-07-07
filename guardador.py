import csv
import os

CSV_FILE = 'humedad_datos.csv'

def init_csv():
    """Crea el CSV con cabecera si aún no existe."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'raw', 'humidity_pct'])

def append_csv(timestamp, raw, pct):
    """Agrega una fila al CSV."""
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, raw, pct])
