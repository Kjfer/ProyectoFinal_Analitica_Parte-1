from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
import os
import requests
import json
import threading
import time
from bot.utils import obtener_ultimo_pct


ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
NUMERO_USUARIO = os.getenv("USER_NUMBER")

# Si tienes funciones para obtener datos del sensor:
# from .sensor_utils import obtener_datos

def enviar_mensaje(mensaje, telefono):
    url = f'https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages'
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "messaging_product": "whatsapp",
        "to": telefono,
        "type": "text",
        "text": {"body": mensaje}
    }
    response = requests.post(url, headers=headers, json=data)
    print("📤 WhatsApp:", response.status_code, response.text)

@csrf_exempt
def webhook(request):
    if request.method == 'GET':
        mode = request.GET.get('hub.mode')
        token = request.GET.get('hub.verify_token')
        challenge = request.GET.get('hub.challenge')
        if mode == "subscribe" and token == VERIFY_TOKEN:
            return HttpResponse(challenge)
        return HttpResponse('Token inválido', status=403)

    if request.method == 'POST':
        data = json.loads(request.body)
        if data.get('entry'):
            for entrada in data['entry']:
                for cambio in entrada['changes']:
                    valor = cambio['value']
                    if 'messages' in valor:
                        mensaje = valor['messages'][0]
                        texto = mensaje['text']['body'].lower()
                        telefono = mensaje['from']

                        # Aquí puedes usar tu función obtener_datos()
                        # datos = obtener_datos()
                        # pct = datos.get("pct")

                        if "filtración" in texto or "hay filtraciones" in texto:
                            pct = obtener_ultimo_pct()
                            if pct is not None and pct > 60:
                                enviar_mensaje(f"⚠️ Filtración detectada. Revisar zona afectada. Porcentaje de humenda: {pct} %.", telefono)
                            else:
                                enviar_mensaje(f"✅ No se detectan filtraciones. Porcentaje de humenda: {pct} %.", telefono)

                        elif "sensor" in texto:
                            enviar_mensaje("Uso un sensor de humedad de pared.", telefono)

                        elif "registr" in texto:
                            enviar_mensaje("✅ Sí, estoy guardando todos los eventos.", telefono)

                        elif "nivel crítico" in texto or "crítico" in texto:
                            enviar_mensaje("🔴 Se considera crítico si supera el 60% de humedad.", telefono)

                        elif "hola" in texto or "inicio" in texto:
                            enviar_mensaje("Hola 👋 ¿Deseas saber si hay filtraciones?", telefono)

                        elif "no" in texto:
                            enviar_mensaje("👌 Estoy aquí si detectas problemas.", telefono)

                        elif "sí" in texto or "si" in texto:
                            enviar_mensaje("🔎 Consultando sensor...", telefono)
                            pct = obtener_ultimo_pct()
                            if pct is not None and pct > 60:
                                enviar_mensaje(f"⚠️ Filtración detectada. Revisar zona afectada. Porcentaje de humenda: {pct} %.", telefono)
                            else:
                                enviar_mensaje(f"✅ No se detectan filtraciones. Porcentaje de humenda: {pct} %.", telefono)
        return JsonResponse({'status': 'ok'})
    
def monitorear_sensor():
    while True:
        pct = obtener_ultimo_pct()
        if pct is not None and pct > 60:
            enviar_mensaje("⚠️ Filtración detectada. Revisar zona afectada.", NUMERO_USUARIO)
        time.sleep(30)

def enviar_saludo_inicial():
    enviar_mensaje("👋 ¡Hola! El bot de filtraciones está activo y listo para ayudarte.", NUMERO_USUARIO)

if os.environ.get("RUN_MAIN") == "true":
    enviar_saludo_inicial()
    threading.Thread(target=monitorear_sensor, daemon=True).start()