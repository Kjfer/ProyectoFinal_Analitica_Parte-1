# main_bot.py
import threading
import time
import os
import requests
from flask import Flask, request
from dotenv import load_dotenv

from lector_compartido import iniciar_lector, obtener_datos

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
NUMERO_USUARIO = os.getenv("USER_NUMBER")

app = Flask(__name__)

def enviar_mensaje(mensaje, telefono):
    url = f'https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages'
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

# 👋 Enviar saludo inicial
def enviar_saludo_inicial():
    mensaje = "Hola 👋 ¿Deseas saber si hay filtraciones?"
    enviar_mensaje(mensaje, NUMERO_USUARIO)

# 🔁 Monitorear sensor en segundo plano
def monitorear_sensor():
    while True:
        datos = obtener_datos()
        pct = datos.get("pct")
        if pct is not None and pct > 60:
            enviar_mensaje("⚠️ Filtración detectada. Revisar zona afectada.", NUMERO_USUARIO)
        time.sleep(10)

@app.route('/webhook', methods=['GET'])
def verificar():
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    if token == VERIFY_TOKEN:
        return challenge
    return 'Token inválido', 403

@app.route('/webhook', methods=['POST'])
def recibir_mensaje():
    data = request.get_json()
    if data.get('entry'):
        for entrada in data['entry']:
            for cambio in entrada['changes']:
                valor = cambio['value']
                if 'messages' in valor:
                    mensaje = valor['messages'][0]
                    texto = mensaje['text']['body'].lower()
                    telefono = mensaje['from']

                    if "filtración" in texto or "hay filtraciones" in texto:
                        datos = obtener_datos()
                        pct = datos.get("pct")
                        if pct is not None and pct > 60:
                            enviar_mensaje("⚠️ Filtración detectada. Revisar zona afectada.", telefono)
                        else:
                            enviar_mensaje("✅ No se detectan filtraciones.", telefono)

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
                        datos = obtener_datos()
                        pct = datos.get("pct")
                        if pct is not None and pct > 60:
                            enviar_mensaje("⚠️ Filtración detectada. Revisar zona afectada.", telefono)
                        else:
                            enviar_mensaje("✅ No se detectan filtraciones.", telefono)

    return 'ok', 200

if __name__ == '__main__':
    # 🔄 Iniciar lector de sensor (grafica, guarda, comparte)
    iniciar_lector()

    # 🧵 Iniciar hilo de monitoreo para alertas automáticas
    threading.Thread(target=monitorear_sensor, daemon=True).start()

    # 👋 Saludo inicial
    enviar_saludo_inicial()

    # 🚀 Iniciar servidor Flask
    app.run(port=5000)
