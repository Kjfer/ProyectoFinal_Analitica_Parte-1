from django.utils import timezone
from datetime import timedelta
from django.shortcuts import render
from django.http import JsonResponse
from .models import RegistroHumedad
from django.db.models import Avg, Max, Min
from django.utils.timezone import now
from django.utils.timezone import localtime
from django.http import HttpResponse
import csv

#----------------------------------------------------------------------
# Devuelve los últimos 60 registros de humedad en formato JSON (para gráfico en tiempo real)
def humedad_evolucion_json(request):
    datos = RegistroHumedad.objects.order_by('-timestamp')[:60][::-1]
    lista = [
        {"timestamp": localtime(r.timestamp).strftime("%Y-%m-%d %H:%M:%S"), "humedad": r.humedad}
        for r in datos
    ]
    return JsonResponse(lista, safe=False)
#----------------------------------------------------------------------
# Devuelve el estado del sensor (conectado/desconectado)
def estado_sensor(request):
    ahora = timezone.now()
    conectado = RegistroHumedad.objects.filter(timestamp__gte=ahora - timedelta(seconds=10)).exists()
    return JsonResponse({'conectado': conectado})

#----------------------------------------------------------------------
# Devuelve el tiempo (en segundos) desde el último registro crítico (humedad > 70%)
def tiempo_desde_ultimo_critico(request):
    # Considerar humedad crítica > 70%, ya que eso indica tu comentario
    ultimo = RegistroHumedad.objects.filter(humedad__gt=70).order_by('-timestamp').first()
    if ultimo:
        ahora = timezone.now()
        delta = ahora - ultimo.timestamp
        total_segundos = int(delta.total_seconds())

        # Convertir a formato HH:MM:SS
        tiempo_formateado = str(timedelta(seconds=total_segundos))
    else:
        tiempo_formateado = None

    return JsonResponse({'tiempo_desde_critico': tiempo_formateado})
#----------------------------------------------------------------------
# Devuelve el historial de humedad por día (para el botón de historial)
def historial_por_dia(request):
    fecha = request.GET.get('fecha')
    datos = []
    if fecha:
        datos = RegistroHumedad.objects.filter(timestamp__date=fecha).order_by('timestamp')
    return render(request, 'dashboard/historial.html', {'datos': datos, 'fecha': fecha})

#----------------------------------------------------------------------
# Vista principal del dashboard (todo en dashboard.html)
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

#----------------------------------------------------------------------
# Devuelve estadísticas generales de humedad (máximo, mínimo, promedio, actual)
def estadisticas_humedad(request):
    hoy = timezone.localdate()
    datos = RegistroHumedad.objects.filter(timestamp__date=hoy)
    estadisticas = datos.aggregate(
        promedio=Avg('humedad'),
        minimo=Min('humedad'),
        maximo=Max('humedad'),
    )
    ultimo = datos.order_by('-timestamp').first()
    actual = ultimo.humedad if ultimo else 0
    estadisticas['actual'] = actual
    return JsonResponse(estadisticas)

#----------------------------------------------------------------------
# Devuelve el historial de humedad por día (para el botón de historial)
def historial_completo(request):
    """
    Reporte histórico diario, mensual, anual o por rango de fechas.
    Devuelve: datos, máximo, mínimo, promedio, último valor y lista para gráfico.
    """
    datos = RegistroHumedad.objects.all().order_by('timestamp')
    fecha = request.GET.get('fecha')
    mes = request.GET.get('mes')
    anio = request.GET.get('anio')
    fecha_inicio = request.GET.get('fecha_inicio')
    fecha_fin = request.GET.get('fecha_fin')

    # Filtrado
    if fecha:
        datos = datos.filter(timestamp__date=fecha)
        filtro = f"Fecha: {fecha}"
    elif mes:
        year, month = mes.split('-')
        datos = datos.filter(timestamp__year=year, timestamp__month=month)
        filtro = f"Mes: {mes}"
    elif anio:
        datos = datos.filter(timestamp__year=anio)
        filtro = f"Año: {anio}"
    elif fecha_inicio and fecha_fin:
        datos = datos.filter(timestamp__date__gte=fecha_inicio, timestamp__date__lte=fecha_fin)
        filtro = f"Rango: {fecha_inicio} a {fecha_fin}"
    else:
        filtro = "Todo el historial"

    # Indicadores del periodo filtrado
    stats = datos.aggregate(
        maximo=Max('humedad'),
        minimo=Min('humedad'),
        promedio=Avg('humedad'),
    )
    ultimo = datos.order_by('-timestamp').first()
    stats['ultimo'] = ultimo.humedad if ultimo else None

    if fecha:  # Solo un día
        label_format = "%H:%M:%S"
    else:      # Varios días, meses, años o rango
        label_format = "%Y-%m-%d %H:%M:%S"

    grafico = [
        {"timestamp": localtime(d.timestamp).strftime(label_format), "humedad": d.humedad}
        for d in datos
    ]

    return render(request, 'dashboard/historial_completo.html', {
        'datos': datos,
        'stats': stats,
        'filtro': filtro,
        'grafico': grafico,
        'fecha': fecha,
        'mes': mes,
        'anio': anio,
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin,
    })

#----------------------------------------------------------------------
# Exporta el historial completo a un archivo CSV
def descargar_historial_csv(request):
    # Filtra los datos igual que en historial_completo
    fecha = request.GET.get('fecha')
    mes = request.GET.get('mes')
    anio = request.GET.get('anio')
    fecha_inicio = request.GET.get('fecha_inicio')
    fecha_fin = request.GET.get('fecha_fin')

    datos = RegistroHumedad.objects.all()
    if fecha:
        datos = datos.filter(timestamp__date=fecha)
    elif mes:
        datos = datos.filter(timestamp__month=mes.split('-')[1], timestamp__year=mes.split('-')[0])
    elif anio:
        datos = datos.filter(timestamp__year=anio)
    elif fecha_inicio and fecha_fin:
        datos = datos.filter(timestamp__date__range=[fecha_inicio, fecha_fin])

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="historial_humedad.csv"'
    writer = csv.writer(response)
    writer.writerow(['Fecha y Hora', 'Humedad (%)'])
    for d in datos:
        writer.writerow([d.timestamp, d.humedad])
    return response
#----------------------------------------------------------------------
# chatbot con OpenRouter
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chatbot import responder_chatbot

@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        pregunta = request.POST.get("pregunta")
        respuesta = responder_chatbot(pregunta or "")
        return JsonResponse({"respuesta": respuesta})
    return JsonResponse({"error": "Solo POST permitido"}, status=400)

def chatbot_page(request):
    # Renderiza tu plantilla dashboard.html, que ya tiene el formulario de chat
    return render(request, 'dashboard/dashboard.html')
