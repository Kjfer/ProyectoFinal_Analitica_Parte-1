from django.core.management.base import BaseCommand
from bot.lector_serial import iniciar_puerto, leer_datos
from dashboard.models import RegistroHumedad
from datetime import datetime
import time

class Command(BaseCommand):
    help = 'Inicia la lectura continua del sensor y guarda en la base de datos'

    def handle(self, *args, **kwargs):
        ser = iniciar_puerto()
        if not ser:
            self.stdout.write(self.style.ERROR("No se pudo abrir el puerto serial"))
            return

        self.stdout.write(self.style.SUCCESS("Lector iniciado. Presiona Ctrl+C para detener."))
        try:
            while True:
                datos = leer_datos(ser)
                if datos:
                    raw, pct = datos
                    RegistroHumedad.objects.create(raw=raw, humedad=pct)
                    print(f"{datetime.now()} → raw:{raw}  pct:{pct}%")
                time.sleep(1)
        except KeyboardInterrupt:
            print("Lector detenido.")
        finally:
            ser.close()