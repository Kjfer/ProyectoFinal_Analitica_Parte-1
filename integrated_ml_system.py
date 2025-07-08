# integrated_ml_system.py
"""
Sistema integrado que combina el análisis de ML con el chatbot de WhatsApp
para DryWall Alert con detección inteligente de anomalías

Este módulo implementa la integración en tiempo real del sistema:
1. Carga modelos de ML entrenados previamente
2. Recibe datos en tiempo real del sensor de humedad
3. Aplica algoritmos de ML para detectar anomalías
4. Genera alertas inteligentes con análisis de confianza
5. Se integra con el sistema de mensajería WhatsApp

El objetivo es reemplazar alertas simples por umbral con un sistema
inteligente que reduce falsas alarmas y mejora la detección.
"""

# Librerías para análisis de datos y machine learning
import pandas as pd  # Manipulación de datos estructurados
import numpy as np   # Operaciones matemáticas y arrays
import pickle        # Serialización para guardar/cargar modelos entrenados
import time          # Control de tiempo para monitoreo continuo
from datetime import datetime  # Manejo de timestamps y fechas
import os            # Interacción con sistema operativo (variables de entorno)

# Algoritmos de ML específicos utilizados en el sistema integrado
from sklearn.ensemble import RandomForestClassifier, IsolationForest
# - RandomForest: Mejor modelo supervisado según análisis comparativo
# - IsolationForest: Mejor modelo de detección de anomalías no supervisado

from sklearn.preprocessing import StandardScaler  # Normalización de datos
from sklearn.model_selection import train_test_split  # División de datos
import warnings
warnings.filterwarnings('ignore')  # Suprimir advertencias para output limpio

class SmartDryWallDetector:
    """
    Detector inteligente de filtraciones usando Machine Learning en tiempo real.
    
    Esta clase encapsula toda la lógica de detección inteligente:
    - Entrena modelos ML con datos históricos
    - Guarda/carga modelos para uso persistente
    - Analiza lecturas en tiempo real del sensor
    - Combina múltiples algoritmos para mayor precisión
    - Genera alertas contextualizadas con niveles de confianza
    
    Flujo de funcionamiento:
    1. Entrenamiento inicial con datos históricos (train_models)
    2. Persistencia de modelos entrenados (save/load_models)
    3. Análisis en tiempo real de cada lectura (predict_anomaly)
    4. Generación de alertas inteligentes (generate_alert_message)
    5. Monitoreo continuo integrado (continuous_monitoring)
    
    Algoritmos utilizados:
    - RandomForest: Clasificación supervisada principal
    - IsolationForest: Detección de anomalías no supervisada
    - Combinación híbrida para mayor robustez
    """
    
    def __init__(self, data_file='humedad_datos.csv'):
        """
        Inicializa el detector inteligente de filtraciones.
        
        Args:
            data_file (str): Archivo CSV con datos históricos del sensor
            
        Atributos:
            data_file: Ruta a datos históricos para entrenamiento
            model: Clasificador Random Forest (modelo supervisado principal)
            anomaly_detector: Isolation Forest (detector de anomalías no supervisado)
            scaler: Normalizador de características para consistencia
            is_trained: Flag que indica si los modelos están entrenados
            threshold_basic: Umbral básico de respaldo (50% humedad)
        """
        self.data_file = data_file
        
        # Modelos de Machine Learning
        self.model = None              # Random Forest para clasificación
        self.anomaly_detector = None   # Isolation Forest para anomalías
        self.scaler = StandardScaler() # Normalizador de datos
        
        # Estado del sistema
        self.is_trained = False        # ¿Modelos entrenados?
        self.threshold_basic = 50      # Umbral de respaldo (humedad %)
        
        print("🤖 Smart DryWall Detector inicializado")
        print(f"📂 Datos de entrenamiento: {data_file}")
        print(f"⚙️ Umbral básico de respaldo: {self.threshold_basic}%")
        
    def train_models(self):
        """
        Entrena los modelos de ML con datos históricos del sensor.
        
        Este método implementa el pipeline completo de entrenamiento:
        1. Carga datos históricos desde CSV
        2. Realiza feature engineering (extracción de características)
        3. Prepara datos (normalización, división)
        4. Entrena dos modelos complementarios:
           - Random Forest: Supervisado (usa etiquetas conocidas)
           - Isolation Forest: No supervisado (detecta patrones anómalos)
        5. Evalúa el rendimiento en datos de prueba
        6. Persiste modelos para uso futuro
        
        La combinación de ambos modelos permite:
        - Mayor robustez en la detección
        - Validación cruzada entre enfoques
        - Reducción de falsas alarmas
        """
        print("🧠 Entrenando modelos de Machine Learning...")
        print("📊 Cargando datos históricos para aprendizaje...")
        
        # Cargar y preparar datos históricos
        df = pd.read_csv(self.data_file)
        print(f"✅ {len(df)} registros históricos cargados")
        
        # Feature Engineering: Crear características temporales
        # El análisis temporal puede revelar patrones de filtración
        # Ejemplo: filtraciones más comunes en ciertos horarios
        df['hour'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S').dt.hour
        df['minute'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S').dt.minute
        
        # Crear variable objetivo basada en umbral validado
        # 50% de humedad es el punto crítico según estándares de construcción
        df['is_anomaly'] = (df['humidity_pct'] > self.threshold_basic).astype(int)
        
        print(f"🎯 Casos normales: {(df['is_anomaly'] == 0).sum()}")
        print(f"🚨 Casos anómalos: {(df['is_anomaly'] == 1).sum()}")
        print(f"📊 Tasa de anomalías: {df['is_anomaly'].mean():.2%}")
        
        # Preparar características para entrenamiento
        # Incluimos tanto valores raw como porcentajes y contexto temporal
        feature_columns = ['raw', 'humidity_pct', 'hour', 'minute']
        X = df[feature_columns]  # Matriz de características
        y = df['is_anomaly']     # Vector de etiquetas objetivo
        
        # Normalización crítica para algoritmos ML
        # Asegura que todas las características tengan la misma escala
        X_scaled = self.scaler.fit_transform(X)
        print("📏 Datos normalizados (media=0, desviación=1)")
        
        # División estratificada para mantener proporción de clases
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.3,      # 30% para evaluación
            random_state=42,    # Reproducibilidad
            stratify=y          # Mantener proporción de anomalías
        )
        
        print(f"📚 Datos entrenamiento: {len(X_train)} muestras")
        print(f"🧪 Datos evaluación: {len(X_test)} muestras")
        
        # ENTRENAMIENTO MODELO 1: Random Forest (Supervisado)
        # Aprende de ejemplos etiquetados históricos
        print("\n🌳 Entrenando Random Forest (clasificación supervisada)...")
        self.model = RandomForestClassifier(
            n_estimators=100,     # 100 árboles para robustez
            random_state=42,      # Reproducibilidad
            max_depth=10,         # Evitar overfitting
            min_samples_split=5   # Mínimo para dividir nodos
        )
        self.model.fit(X_train, y_train)
        
        # ENTRENAMIENTO MODELO 2: Isolation Forest (No supervisado)
        # Detecta patrones anómalos sin usar etiquetas
        print("🔍 Entrenando Isolation Forest (detección de anomalías)...")
        self.anomaly_detector = IsolationForest(
            contamination=0.1,    # Esperamos ~10% de anomalías
            random_state=42,      # Reproducibilidad
            n_estimators=100      # 100 árboles de aislamiento
        )
        self.anomaly_detector.fit(X_train)
        
        # Evaluación en datos de prueba
        accuracy_rf = self.model.score(X_test, y_test)
        print(f"\n📊 Evaluación en datos de prueba:")
        print(f"   🎯 Random Forest Accuracy: {accuracy_rf:.4f}")
        
        # Evaluar detector de anomalías
        anomaly_predictions = self.anomaly_detector.predict(X_test)
        anomaly_predictions = np.where(anomaly_predictions == -1, 1, 0)
        anomaly_accuracy = np.mean(anomaly_predictions == y_test)
        print(f"   🔍 Isolation Forest Accuracy: {anomaly_accuracy:.4f}")
        
        # Marcar como entrenado y guardar
        self.is_trained = True
        self.save_models()
        
        print("✅ Entrenamiento completado exitosamente")
        print("💾 Modelos guardados para uso futuro")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Entrenar Random Forest (mejor modelo supervisado)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Entrenar Isolation Forest (detección de anomalías)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_train)
        
        # Evaluar
        accuracy = self.model.score(X_test, y_test)
        print(f"✅ Modelo entrenado con accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        
        # Guardar modelos
        self.save_models()
        
    def save_models(self):
        """
        Guarda los modelos entrenados
        """
        with open('ml_models.pkl', 'wb') as f:
            pickle.dump({
                'classifier': self.model,
                'anomaly_detector': self.anomaly_detector,
                'scaler': self.scaler
            }, f)
        print("💾 Modelos guardados en ml_models.pkl")
        
    def load_models(self):
        """
        Carga modelos pre-entrenados
        """
        try:
            with open('ml_models.pkl', 'rb') as f:
                models = pickle.load(f)
                self.model = models['classifier']
                self.anomaly_detector = models['anomaly_detector']
                self.scaler = models['scaler']
                self.is_trained = True
            print("✅ Modelos cargados exitosamente")
            return True
        except FileNotFoundError:
            print("⚠️ No se encontraron modelos pre-entrenados")
            return False
            
    def predict_anomaly(self, raw_value, humidity_pct, hour=None, minute=None):
        """
        Predice si una lectura del sensor indica anomalía usando ML inteligente.
        
        Este es el corazón del sistema de detección en tiempo real. Combina
        múltiples enfoques de ML para una detección más robusta y confiable.
        
        Proceso de predicción:
        1. Verificar disponibilidad de modelos entrenados
        2. Preparar características de entrada
        3. Ejecutar predicción con Random Forest (supervisado)
        4. Ejecutar detección con Isolation Forest (no supervisado)
        5. Combinar resultados con lógica de consenso
        6. Calcular nivel de confianza de la predicción
        
        Args:
            raw_value (int): Valor crudo del sensor de humedad
            humidity_pct (float): Porcentaje de humedad calculado
            hour (int, optional): Hora actual (0-23)
            minute (int, optional): Minuto actual (0-59)
            
        Returns:
            tuple: (is_anomaly, method, confidence, anomaly_score)
            - is_anomaly (bool): ¿Se detectó anomalía?
            - method (str): Método de detección utilizado
            - confidence (float): Nivel de confianza (0.0-1.0)
            - anomaly_score (float): Score numérico de anomalía
        
        Lógica de consenso:
        - Ambos modelos detectan → ALTA CONFIANZA
        - Solo supervisado detecta → MEDIA CONFIANZA  
        - Solo no supervisado detecta → BAJA CONFIANZA
        - Ninguno detecta → NORMAL
        """
        # Verificar si los modelos están disponibles
        if not self.is_trained:
            print("⚠️ Modelos no entrenados, intentando cargar...")
            if not self.load_models():
                print("❌ Modelos no disponibles, usando detección básica")
                is_basic_anomaly = humidity_pct > self.threshold_basic
                return is_basic_anomaly, "Umbral básico (50%)", 0.6, 0.0
        
        # Preparar características para predicción
        # Usar tiempo actual si no se proporciona
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour if hour is None else hour
            minute = now.minute if minute is None else minute
            
        # Crear vector de características idéntico al entrenamiento
        features = np.array([[raw_value, humidity_pct, hour, minute]])
        features_scaled = self.scaler.transform(features)  # Aplicar misma normalización
        
        # ============= PREDICCIÓN CON MODELO SUPERVISADO =============
        # Random Forest da probabilidades de clase
        prob_anomaly = self.model.predict_proba(features_scaled)[0][1]  # Probabilidad de anomalía
        is_anomaly_classifier = self.model.predict(features_scaled)[0]   # Predicción binaria
        
        # ============= DETECCIÓN CON MODELO NO SUPERVISADO =============
        # Isolation Forest da score de anomalía y predicción binaria
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly_detector = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # ============= LÓGICA DE CONSENSO INTELIGENTE =============
        confidence = prob_anomaly  # Confianza base del clasificador
        is_anomaly = False
        method = "ML Consenso"
        
        if is_anomaly_classifier and is_anomaly_detector:
            # Ambos detectan anomalía → MÁXIMA CONFIANZA
            is_anomaly = True
            method = "ML Alto Riesgo (Consenso)"
            confidence = min(prob_anomaly + 0.2, 1.0)  # Boost de confianza
            
        elif is_anomaly_classifier:
            # Solo supervisado detecta → CONFIANZA MEDIA
            is_anomaly = True
            method = "ML Clasificador"
            confidence = prob_anomaly
            
        elif is_anomaly_detector:
            # Solo no supervisado detecta → CONFIANZA BAJA/MEDIA
            is_anomaly = True
            method = "ML Detector Anomalías"
            confidence = 0.7  # Confianza moderada
            
        else:
            # Ninguno detecta → NORMAL
            is_anomaly = False
            method = "ML Normal"
            confidence = 1.0 - prob_anomaly  # Confianza en normalidad
            
        return is_anomaly, method, confidence, anomaly_score
    
    def get_risk_level(self, humidity_pct, confidence=0.5):
        """
        Determina el nivel de riesgo
        """
        if humidity_pct < 20:
            return "🟢 BAJO", "Ambiente seco, sin riesgo"
        elif humidity_pct < 40:
            return "🟡 NORMAL", "Humedad en rango normal"
        elif humidity_pct < 60:
            return "🟠 ALTO", "Humedad elevada, monitorear"
        else:
            return "🔴 CRÍTICO", "Posible filtración detectada"
    
    def generate_alert_message(self, raw_value, humidity_pct):
        """
        Genera mensaje de alerta inteligente con análisis ML contextualizado.
        
        Esta función crea mensajes personalizados basados en:
        - Predicción ML (anomalía detectada o no)
        - Nivel de confianza del algoritmo
        - Contexto de riesgo según humedad
        - Recomendaciones específicas para la situación
        
        Los mensajes incluyen:
        - Emoji descriptivo según severidad
        - Datos técnicos del sensor
        - Análisis ML detallado (método y confianza)
        - Recomendaciones contextualizadas
        - Nivel de urgencia de la respuesta
        
        Args:
            raw_value (int): Valor crudo del sensor
            humidity_pct (float): Porcentaje de humedad
            
        Returns:
            tuple: (message, is_anomaly)
            - message (str): Mensaje formateado para WhatsApp
            - is_anomaly (bool): Indica si hay anomalía detectada
        """
        # Ejecutar análisis ML completo
        is_anomaly, method, confidence, anomaly_score = self.predict_anomaly(raw_value, humidity_pct)
        risk_level, risk_description = self.get_risk_level(humidity_pct, confidence)
        
        if is_anomaly:
            # ============= MENSAJE DE ALERTA =============
            message = f"🚨 ALERTA DE FILTRACIÓN DETECTADA\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            # Datos técnicos del sensor
            message += f"📊 DATOS DEL SENSOR:\n"
            message += f"   • Humedad: {humidity_pct:.1f}%\n"
            message += f"   • Valor raw: {raw_value}\n"
            message += f"   • Timestamp: {datetime.now().strftime('%H:%M:%S')}\n"
            message += f"   • Nivel de riesgo: {risk_level}\n\n"
            
            # Análisis de Machine Learning
            message += f"🧠 ANÁLISIS INTELIGENTE:\n"
            message += f"   • Método detección: {method}\n"
            message += f"   • Confianza ML: {confidence:.1%}\n"
            message += f"   • Score anomalía: {anomaly_score:.3f}\n"
            
            # Interpretación del score
            if anomaly_score < -0.3:
                score_interpretation = "Muy anómalo"
            elif anomaly_score < -0.1:
                score_interpretation = "Moderadamente anómalo"
            else:
                score_interpretation = "Ligeramente anómalo"
            message += f"   • Interpretación: {score_interpretation}\n\n"
            
            # Recomendaciones contextualizadas
            message += f"💡 RECOMENDACIÓN:\n"
            message += f"   {risk_description}\n\n"
            
            # Nivel de urgencia
            if confidence > 0.8:
                urgency = "🔴 URGENTE - Revisar inmediatamente"
            elif confidence > 0.6:
                urgency = "� MODERADO - Revisar en las próximas horas"
            else:
                urgency = "🟡 PRECAUCIÓN - Monitorear de cerca"
                
            message += f"⚡ URGENCIA: {urgency}\n"
            message += f"🔧 Inspeccionar zona del sensor ahora"
            
        else:
            # ============= MENSAJE DE ESTADO NORMAL =============
            message = f"✅ SISTEMA DRYWALL NORMAL\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📊 Humedad: {humidity_pct:.1f}% | Raw: {raw_value}\n"
            message += f"🧠 Análisis ML: {method}\n"
            message += f"🎯 Confianza: {confidence:.1%}\n"
            message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            
        return message, is_anomaly
    
    def continuous_monitoring(self, get_sensor_data_func, send_message_func):
        """
        Monitoreo continuo con ML integrado
        """
        print("🔄 Iniciando monitoreo inteligente...")
        
        last_alert_time = 0
        alert_cooldown = 300  # 5 minutos entre alertas
        
        while True:
            try:
                # Obtener datos del sensor
                data = get_sensor_data_func()
                if data is None:
                    continue
                    
                raw_value = data.get('raw', 0)
                humidity_pct = data.get('pct', 0)
                
                # Generar mensaje usando ML
                message, is_anomaly = self.generate_alert_message(raw_value, humidity_pct)
                
                # Enviar alerta si es necesario
                current_time = datetime.now().timestamp()
                if is_anomaly and (current_time - last_alert_time) > alert_cooldown:
                    send_message_func(message)
                    last_alert_time = current_time
                    print(f"🚨 Alerta enviada: {humidity_pct}%")
                else:
                    print(f"📊 Normal: {humidity_pct}% (ML: {message.split('ML: ')[1] if 'ML: ' in message else 'N/A'})")
                
                time.sleep(10)  # Esperar 10 segundos
                
            except Exception as e:
                print(f"❌ Error en monitoreo: {e}")
                time.sleep(5)

def integrate_with_existing_bot():
    """
    Función para integrar con el bot de WhatsApp existente
    """
    print("🔗 Integrando ML con sistema existente...")
    
    # Importar funciones del sistema existente
    try:
        from lector_compartido import obtener_datos
        from main_bot import enviar_mensaje
        detector = SmartDryWallDetector()
        
        # Entrenar modelos si no existen
        if not detector.load_models():
            detector.train_models()
        
        # Función wrapper para obtener datos
        def get_sensor_data():
            return obtener_datos()
        
        # Función wrapper para enviar mensaje
        def send_alert(message):
            numero_usuario = os.getenv("USER_NUMBER")
            if numero_usuario:
                enviar_mensaje(message, numero_usuario)
            else:
                print("📱 Mensaje de alerta:", message)
        
        # Iniciar monitoreo inteligente
        detector.continuous_monitoring(get_sensor_data, send_alert)
        
    except ImportError as e:
        print(f"⚠️ No se pudo importar módulos existentes: {e}")
        print("💡 Ejecutar desde el directorio del proyecto principal")

def demo_ml_system():
    """
    Demostración del sistema ML
    """
    print("🎯 DEMO - Sistema ML para DryWall Alert")
    print("=" * 50)
    
    detector = SmartDryWallDetector()
    
    # Entrenar modelos
    detector.train_models()
    
    # Casos de prueba
    test_cases = [
        {"raw": 300, "humidity": 15, "description": "Ambiente seco"},
        {"raw": 400, "humidity": 35, "description": "Humedad normal"},
        {"raw": 500, "humidity": 55, "description": "Humedad alta"},
        {"raw": 325, "humidity": 65, "description": "Posible filtración"},
        {"raw": 200, "humidity": 80, "description": "Filtración severa"}
    ]
    
    print("\n🧪 Probando casos de ejemplo:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Caso {i}: {case['description']} ---")
        message, is_anomaly = detector.generate_alert_message(case['raw'], case['humidity'])
        print(message)
        print(f"🎯 Anomalía detectada: {'SÍ' if is_anomaly else 'NO'}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_ml_system()
    elif len(sys.argv) > 1 and sys.argv[1] == "integrate":
        integrate_with_existing_bot()
    else:
        print("🏠 DRYWALL ALERT - Sistema ML Integrado")
        print("\nUso:")
        print("  python integrated_ml_system.py demo      # Ejecutar demostración")
        print("  python integrated_ml_system.py integrate # Integrar con sistema existente")
        demo_ml_system()
