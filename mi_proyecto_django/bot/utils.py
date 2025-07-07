from dashboard.models import RegistroHumedad

def obtener_ultimo_pct():
    ultimo = RegistroHumedad.objects.order_by('-timestamp').first()
    if ultimo:
        return ultimo.humedad
    return None