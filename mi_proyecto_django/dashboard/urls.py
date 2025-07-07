from django.urls import path
from . import views

urlpatterns = [
    # Dashboard principal
    path('', views.dashboard_view, name='dashboard'),
    path('humedad-evolucion-json/', views.humedad_evolucion_json, name='humedad_evolucion_json'),
    path('estadisticas-humedad/', views.estadisticas_humedad, name='estadisticas_humedad'),
    path('estado-sensor/', views.estado_sensor, name='estado_sensor'),
    path('tiempo-ultimo-critico/', views.tiempo_desde_ultimo_critico, name='tiempo_desde_ultimo_critico'),
    path('historial/', views.historial_por_dia, name='historial_por_dia'),
    path('historial-completo/', views.historial_completo, name='historial_completo'),
    path('descargar_historial_csv/', views.descargar_historial_csv, name='descargar_historial_csv'),
    path('chat/', views.chatbot_page, name='chatbot_page'),
    path('chatbot_api/', views.chatbot_api, name='chatbot_api'),
]