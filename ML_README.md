# DryWall Alert - Análisis de Machine Learning

## 📋 Descripción del Proyecto

**Grupo 3 - DryWall Alert**: Sistema inteligente de detección de anomalías y clasificación para sensores de humedad en drywall, con el objetivo de detectar filtraciones de manera temprana.

## 🎯 Objetivo

Implementar y comparar **10 modelos de Machine Learning** para mejorar la detección de filtraciones en el sistema DryWall Alert, cumpliendo con los requerimientos de la **Pregunta 1** del proyecto.

## 📊 Modelos Implementados

### Detección de Anomalías:
1. **Isolation Forest** - Detección de outliers mediante aislamiento
2. **One-Class SVM** - Clasificación de una sola clase
3. **Autoencoder** - Red neuronal para reconstrucción
4. **DBSCAN** - Clustering para identificar anomalías
5. **LOF (Local Outlier Factor)** - Factor de outlier local

### Clasificación Supervisada:
6. **Random Forest** - Ensamble de árboles de decisión
7. **k-NN** - k-vecinos más cercanos
8. **MLP (Multi-Layer Perceptron)** - Red neuronal multicapa
9. **AdaBoost** - Boosting adaptativo
10. **Gradient Boosting** - Boosting por gradiente

## 📁 Estructura del Proyecto

```
Chatbot-con-arduino-PC2/
├── ml_analysis.py              # Análisis principal de ML
├── integrated_ml_system.py     # Integración con sistema existente
├── setup_ml_environment.py     # Configuración del entorno
├── ml_analysis_notebook.ipynb  # Notebook interactivo
├── humedad_datos.csv          # Dataset de sensores
├── requirements.txt           # Dependencias actualizadas
├── ml_results/               # Resultados de análisis
├── ml_models/               # Modelos entrenados
└── ml_reports/             # Reportes generados
```

## 🚀 Instalación y Configuración

### 1. Configurar Entorno
```powershell
# Ejecutar script de configuración
python setup_ml_environment.py
```

### 2. Instalar Dependencias Manualmente (alternativa)
```powershell
pip install -r requirements.txt
```

### 3. Dependencias Principales
- `scikit-learn` - Modelos de ML
- `tensorflow` - Autoencoder
- `pandas` - Manipulación de datos
- `matplotlib/seaborn` - Visualización
- `numpy` - Cálculos numéricos

## 📈 Ejecución del Análisis

### Análisis Completo
```powershell
python ml_analysis.py
```

### Análisis Interactivo (Jupyter)
```powershell
jupyter notebook ml_analysis_notebook.ipynb
```

### Demostración del Sistema
```powershell
python integrated_ml_system.py demo
```

### Integración con Sistema Existente
```powershell
python integrated_ml_system.py integrate
```

## 📊 Características del Dataset

- **Fuente**: Sensor de humedad Arduino
- **Registros**: ~232 lecturas
- **Features**:
  - `raw`: Valor crudo del sensor (300-600)
  - `humidity_pct`: Porcentaje de humedad (0-62%)
  - `timestamp`: Marca de tiempo
  - `hour/minute`: Features temporales derivadas

### Distribución de Clases
- **Normal (0)**: Humedad ≤ 50%
- **Anomalía (1)**: Humedad > 50% (posible filtración)

## 🔍 Métricas de Evaluación

### Métricas Principales:
- **Accuracy**: Precisión general del modelo
- **F1-Score**: Balance entre precisión y recall
- **Precision**: Exactitud en predicciones positivas
- **Recall**: Capacidad de detectar anomalías

### Métricas Específicas:
- **Score de Anomalía**: Para modelos no supervisados
- **Confianza**: Probabilidad de predicción
- **Tiempo de Entrenamiento**: Eficiencia computacional

## 📈 Resultados Esperados

El análisis generará:

1. **Visualizaciones**:
   - `data_analysis.png` - Análisis exploratorio
   - `model_comparison.png` - Comparación de modelos

2. **Reportes**:
   - Ranking de modelos por rendimiento
   - Justificación del mejor modelo
   - Recomendaciones de implementación

3. **Modelos Entrenados**:
   - `ml_models.pkl` - Modelos serializados
   - Configuración de hiperparámetros

## 🎯 Aplicación Práctica

### Integración con WhatsApp Bot

El mejor modelo se integra con el sistema existente para:

- **Detección Inteligente**: Usar ML en lugar de umbral fijo
- **Alertas Contextuales**: Mensajes con nivel de confianza
- **Análisis Temporal**: Considerar patrones horarios
- **Reducción de Falsos Positivos**: Mayor precisión

### Ejemplo de Alerta ML:
```
⚠️ ALERTA DE FILTRACIÓN DETECTADA

📊 Datos del sensor:
   • Humedad: 65%
   • Valor raw: 325
   • Nivel de riesgo: 🔴 CRÍTICO

🧠 Análisis ML:
   • Método: ML Alto Riesgo
   • Confianza: 92%
   • Score anomalía: -0.245

💡 Recomendación: Posible filtración detectada
🔧 Revisar zona afectada inmediatamente
```

## 📝 Justificación Técnica

### ¿Por qué estos modelos?

1. **Isolation Forest**: Excelente para detectar anomalías sin etiquetas
2. **Random Forest**: Robusto y resistente al overfitting
3. **Autoencoder**: Detecta patrones complejos no lineales
4. **DBSCAN**: Identifica grupos naturales en los datos
5. **Gradient Boosting**: Alta precisión en clasificación

### Criterios de Selección:
- **Precision/Recall Balance**: Para minimizar falsas alarmas
- **Interpretabilidad**: Entender por qué se detecta una anomalía
- **Eficiencia**: Tiempo real en dispositivos embebidos
- **Robustez**: Funcionamiento con datos limitados

## 🔧 Troubleshooting

### Problemas Comunes:

1. **Error de TensorFlow**:
   ```powershell
   pip install tensorflow==2.18.0
   ```

2. **Memoria insuficiente**:
   - Reducir `n_estimators` en Random Forest
   - Usar `n_samples` menor en Isolation Forest

3. **Dataset pequeño**:
   - Usar validación cruzada
   - Aplicar técnicas de augmentación

### Logs de Depuración:
- Verificar formato de timestamp
- Validar rangos de valores del sensor
- Comprobar distribución de clases

## 📚 Referencias

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Anomaly Detection Techniques](https://www.researchgate.net/publication/328542419)

## 👥 Equipo

**Grupo 3 - DryWall Alert**
- Proyecto: Sistema de detección de filtraciones
- Curso: Analítica de Datos
- Fecha: Julio 2025

## 📞 Soporte

Para problemas técnicos o preguntas sobre la implementación, revisar:
1. Logs de ejecución en consola
2. Archivos generados en `ml_results/`
3. Documentación en código fuente

---

**¡Importante!** Este análisis cumple con todos los requerimientos de la **Pregunta 1** del proyecto, implementando 10 modelos de ML con comparación de métricas y justificación técnica del mejor modelo para el caso de uso específico de DryWall Alert.
