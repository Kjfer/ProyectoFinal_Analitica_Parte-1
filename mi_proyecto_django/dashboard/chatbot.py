# dashboard/chatbot.py

from dotenv import load_dotenv
import os, re, requests
from django.utils.timezone import now, timedelta
from .models import RegistroHumedad
from datetime import datetime

# carga tu .env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"

def obtener_resumen_fecha(fecha_obj):
    qs = RegistroHumedad.objects.filter(timestamp__date=fecha_obj)
    if not qs.exists():
        # Buscar el registro más cercano
        anterior = RegistroHumedad.objects.filter(timestamp__date__lt=fecha_obj).order_by('-timestamp').first()
        posterior = RegistroHumedad.objects.filter(timestamp__date__gt=fecha_obj).order_by('timestamp').first()
        if anterior and posterior:
            # Elegir el más cercano en días
            diff_ant = abs((fecha_obj - anterior.timestamp.date()).days)
            diff_pos = abs((posterior.timestamp.date() - fecha_obj).days)
            mas_cercano = anterior if diff_ant <= diff_pos else posterior
        else:
            mas_cercano = anterior or posterior
        if mas_cercano:
            valores = [mas_cercano.humedad]
            return (
                f"No hay datos exactos para el {fecha_obj}. "
                f"El dato más cercano es del {mas_cercano.timestamp.date()}: "
                f"{mas_cercano.humedad}% de humedad."
            )
        return f"No hay datos de humedad para el {fecha_obj} ni fechas cercanas."
    valores = [r.humedad for r in qs]
    prom = sum(valores) / len(valores)
    return (
        f"Humedad del {fecha_obj}: promedio {prom:.1f}%," 
        f" máximo {max(valores)}%, mínimo {min(valores)}%."
    )

def obtener_resumen_semana():
    desde = now() - timedelta(days=7)
    qs = RegistroHumedad.objects.filter(timestamp__gte=desde)
    if not qs.exists():
        return "No hay datos de humedad en los últimos 7 días."
    valores = [r.humedad for r in qs]
    prom = sum(valores) / len(valores)
    return (
        f"Humedad últimos 7 días: promedio {prom:.1f}%," 
        f" máximo {max(valores)}%, mínimo {min(valores)}%."
    )

def responder_chatbot(pregunta_usuario):
    low = pregunta_usuario.lower()
    resumen = ""
    fecha_consulta = None

    # Comparación entre dos fechas explícitas
    fechas = re.findall(r'(\d{4}-\d{2}-\d{2})', pregunta_usuario)
    if len(fechas) == 2:
        fecha1 = datetime.fromisoformat(fechas[0]).date()
        fecha2 = datetime.fromisoformat(fechas[1]).date()
        resumen1 = obtener_resumen_fecha(fecha1)
        resumen2 = obtener_resumen_fecha(fecha2)
        prompt = (
            "Eres un asistente experto en humedad en paredes de edificaciones. "
            "El umbral máximo recomendado de humedad en pared es 60%. "
            "Si la humedad supera ese valor, advierte sobre posibles riesgos de deterioro, moho o problemas estructurales. "
            "Usa un tono conversacional, amable y profesional. "
            f"Tengo estos datos de humedad:\n{resumen1}\n{resumen2}\n\n"
            f"Por favor, haz una comparación entre ambas fechas y responde a:\n“{pregunta_usuario}”"
        )
        return _llm_response(prompt)

    # Comparación entre "hoy" y "ayer"
    if "hoy" in low and "ayer" in low:
        fecha1 = now().date()
        fecha2 = (now() - timedelta(days=1)).date()
        resumen1 = obtener_resumen_fecha(fecha1)
        resumen2 = obtener_resumen_fecha(fecha2)
        prompt = (
            "Eres un asistente experto en humedad en paredes de edificaciones. "
            "El umbral máximo recomendado de humedad en pared es 60%. "
            "Si la humedad supera ese valor, advierte sobre posibles riesgos de deterioro, moho o problemas estructurales. "
            "Usa un tono conversacional, amable y profesional. "
            f"Tengo estos datos de humedad:\n{resumen1}\n{resumen2}\n\n"
            f"Por favor, haz una comparación entre ambas fechas y responde a:\n“{pregunta_usuario}”"
        )
        return _llm_response(prompt)

    # Consulta por una fecha específica, ayer, hace n días, semana, etc.
    m = re.search(r'(\d{4}-\d{2}-\d{2})', pregunta_usuario)
    if m:
        try:
            fecha_obj = datetime.fromisoformat(m.group(1)).date()
            fecha_consulta = fecha_obj
            resumen = obtener_resumen_fecha(fecha_obj)
        except ValueError:
            resumen = f"No pude interpretar la fecha “{m.group(1)}”."
    elif "ayer" in low:
        fecha_obj = (now() - timedelta(days=1)).date()
        fecha_consulta = fecha_obj
        resumen = obtener_resumen_fecha(fecha_obj)
    else:
        m_dias = re.search(r'(hace|de hace)?\s*(\d+|un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s*d[ií]a[s]?', low)
        if m_dias:
            dias_str = m_dias.group(2)
            mapping = {
                "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
                "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10
            }
            dias = mapping.get(dias_str, int(dias_str) if dias_str.isdigit() else 1)
            fecha_obj = (now() - timedelta(days=dias)).date()
            fecha_consulta = fecha_obj
            resumen = obtener_resumen_fecha(fecha_obj)
        elif "semana" in low or "últimos días" in low:
            resumen = obtener_resumen_semana()
            fecha_consulta = now().date()
        else:
            fecha_obj = now().date()
            fecha_consulta = fecha_obj
            resumen = obtener_resumen_fecha(fecha_obj)

    if "No hay datos" in resumen:
        return resumen

    # Si la pregunta es sobre el umbral máximo, puedes dejar este bloque si quieres respuesta directa
    if re.search(r'umbral m[áa]ximo', pregunta_usuario, re.IGNORECASE):
        umbral = 60
        veces = contar_superaciones_umbral(fecha_consulta, umbral)
        return f"El día {fecha_consulta}, la humedad alcanzó o superó el umbral máximo de {umbral}% un total de {veces} veces."

    # Todo lo demás, pásalo al LLM
    prompt = (
        "Eres un asistente experto en humedad en paredes de edificaciones. "
        "El umbral máximo recomendado de humedad en pared es 60%. "
        "Si la humedad supera ese valor, advierte sobre posibles riesgos de deterioro, moho o problemas estructurales. "
        "Usa un tono conversacional, amable y profesional. "
        f"Tengo estos datos de humedad:\n{resumen}\n\n"
        f"Con base en esos datos, responde a:\n“{pregunta_usuario}”"
    )
    return _llm_response(prompt)

def contar_superaciones_umbral(fecha_obj, umbral=60):
    qs = RegistroHumedad.objects.filter(timestamp__date=fecha_obj)
    if not qs.exists():
        return 0
    return qs.filter(humedad__gte=umbral).count()

def _llm_response(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(URL, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
