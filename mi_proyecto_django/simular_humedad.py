import os
import django
import random
import time
from datetime import datetime

# Configura Django para usar modelos desde este script
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mi_proyecto_django.settings')
django.setup()

from dashboard.models import RegistroHumedad

def simular_datos_humedad():
    while True:
        # Generar un valor aleatorio de humedad entre 30 y 80%
        humedad_simulada = random.randint(30, 80)
        raw_simulado = random.randint(200, 800)
        # Crear registro en la base de datos
        nuevo_registro = RegistroHumedad(
            humedad=humedad_simulada,
            raw=raw_simulado   
        )
        nuevo_registro.save()
        
        print(f"[{datetime.now()}] Guardado humedad: {humedad_simulada}%, raw: {raw_simulado}")
        
        # Esperar 5 segundos antes de la siguiente lectura simulada
        time.sleep(5)

if __name__ == "__main__":
    simular_datos_humedad()
