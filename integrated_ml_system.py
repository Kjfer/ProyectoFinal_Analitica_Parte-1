# integrated_ml_system.py
"""
Sistema integrado que combina el análisis de ML con el chatbot de WhatsApp
para DryWall Alert con detección inteligente de anomalías
"""

import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SmartDryWallDetector:
    def __init__(self, data_file='humedad_datos.csv'):
        """
        Detector inteligente de filtraciones usando Machine Learning
        """
        self.data_file = data_file
        self.model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.threshold_basic = 50  # Umbral básico original
        
    def train_models(self):
        """
        Entrena los modelos de ML con los datos históricos
        """
        print("🧠 Entrenando modelos de Machine Learning...")
        
        # Cargar datos
        df = pd.read_csv(self.data_file)
        
        # Crear features
        df['hour'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S').dt.hour
        df['minute'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S').dt.minute
        df['is_anomaly'] = (df['humidity_pct'] > self.threshold_basic).astype(int)
        
        # Preparar datos
        X = df[['raw', 'humidity_pct', 'hour', 'minute']]
        y = df['is_anomaly']
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
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
        Predice si hay una anomalía usando ML
        """
        if not self.is_trained:
            if not self.load_models():
                print("❌ Modelos no disponibles, usando umbral básico")
                return humidity_pct > self.threshold_basic, "Umbral básico"
        
        # Preparar datos
        if hour is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            
        features = np.array([[raw_value, humidity_pct, hour, minute]])
        features_scaled = self.scaler.transform(features)
        
        # Predicción con clasificador
        prob_anomaly = self.model.predict_proba(features_scaled)[0][1]
        is_anomaly_classifier = self.model.predict(features_scaled)[0]
        
        # Predicción con detector de anomalías
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly_detector = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # Decisión final combinada
        confidence = prob_anomaly
        is_anomaly = is_anomaly_classifier or is_anomaly_detector
        
        method = "ML Combinado"
        if is_anomaly_classifier and is_anomaly_detector:
            method = "ML Alto Riesgo"
            confidence = min(prob_anomaly + 0.2, 1.0)
        elif is_anomaly_classifier:
            method = "ML Clasificador"
        elif is_anomaly_detector:
            method = "ML Detector Anomalías"
            confidence = 0.7
            
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
        Genera mensaje de alerta inteligente
        """
        is_anomaly, method, confidence, anomaly_score = self.predict_anomaly(raw_value, humidity_pct)
        risk_level, risk_description = self.get_risk_level(humidity_pct, confidence)
        
        if is_anomaly:
            message = f"⚠️ ALERTA DE FILTRACIÓN DETECTADA\n\n"
            message += f"📊 Datos del sensor:\n"
            message += f"   • Humedad: {humidity_pct}%\n"
            message += f"   • Valor raw: {raw_value}\n"
            message += f"   • Nivel de riesgo: {risk_level}\n\n"
            message += f"🧠 Análisis ML:\n"
            message += f"   • Método: {method}\n"
            message += f"   • Confianza: {confidence:.2%}\n"
            message += f"   • Score anomalía: {anomaly_score:.3f}\n\n"
            message += f"💡 Recomendación: {risk_description}\n"
            message += f"🔧 Revisar zona afectada inmediatamente"
        else:
            message = f"✅ Sistema normal\n"
            message += f"📊 Humedad: {humidity_pct}% | Raw: {raw_value}\n"
            message += f"🧠 ML: {method} (Confianza: {confidence:.2%})"
            
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
