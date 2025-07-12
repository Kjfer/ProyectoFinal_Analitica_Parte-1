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
    
    def __init__(self, data_file='synthetic_drywall_data_7days.csv'):
        """
        Inicializa el detector inteligente de filtraciones.
        
        ACTUALIZADO: Adaptado para el nuevo dataset sintético de 7 días con
        características expandidas que mejoran significativamente la detección.
        
        Args:
            data_file (str): Archivo CSV con datos históricos del sensor
                           Por defecto usa el nuevo dataset de 7 días
            
        Atributos:
            data_file: Ruta a datos históricos para entrenamiento
            model: Clasificador Random Forest (modelo supervisado principal)
            anomaly_detector: Isolation Forest (detector de anomalías no supervisado)
            scaler: Normalizador de características para consistencia
            is_trained: Flag que indica si los modelos están entrenados
            threshold_basic: Umbral básico de respaldo (50% humedad)
            feature_columns: Lista de características para predicción
        """
        self.data_file = data_file
        
        # Modelos de Machine Learning
        self.model = None              # Random Forest para clasificación
        self.anomaly_detector = None   # Isolation Forest para anomalías
        self.scaler = StandardScaler() # Normalizador de datos
        
        # Estado del sistema
        self.is_trained = False        # ¿Modelos entrenados?
        self.threshold_basic = 50      # Umbral de respaldo (humedad %)
        
        # NUEVO: Características expandidas del dataset sintético
        self.feature_columns = [
            'humidity_pct',           # Humedad principal
            'raw_value',             # Valor crudo del sensor  
            'raw_normalized',        # Valor raw normalizado
            'hour',                  # Hora del día
            'day_of_week',          # Día de semana
            'is_weekend',           # ¿Es fin de semana?
            'is_night',             # ¿Es horario nocturno?
            'humidity_category',     # Categoría de humedad
            'humidity_risk_level',   # Nivel de riesgo calculado
            'sensor_stability',      # Estabilidad del sensor
            'humidity_change',       # Cambio en humedad
            'raw_change'            # Cambio en valor raw
        ]
        
        print("🤖 Smart DryWall Detector inicializado (Dataset 7 días)")
        print(f"📂 Datos de entrenamiento: {data_file}")
        print(f"⚙️ Características ML: {len(self.feature_columns)}")
        print(f"⚙️ Umbral básico de respaldo: {self.threshold_basic}%")
        
    def train_models(self):
        """
        Entrena los modelos de ML con datos históricos del sensor.
        
        ACTUALIZADO: Aprovecha las nuevas características del dataset sintético
        de 7 días para entrenar modelos más precisos y robustos.
        
        Este método implementa el pipeline completo de entrenamiento:
        1. Carga datos históricos enriquecidos (10,080 registros)
        2. Utiliza características pre-calculadas del dataset sintético
        3. Entrena modelos con características expandidas (12 features)
        4. Evalúa rendimiento en datos de prueba
        5. Persiste modelos para uso futuro
        
        Ventajas del nuevo dataset:
        - 13x más datos (10,080 vs ~800 registros)
        - 3x más características (12 vs 4 features)
        - Variables objetivo ya calculadas
        - Características temporales y contextuales avanzadas
        - Datos sintéticos balanceados y representativos
        """
        print("🧠 Entrenando modelos ML con dataset sintético de 7 días...")
        print("📊 Cargando datos históricos enriquecidos...")
        
        # Cargar y preparar datos enriquecidos
        df = pd.read_csv(self.data_file)
        print(f"✅ {len(df):,} registros históricos cargados (7 días)")
        print(f"📅 Periodo: {pd.to_datetime(df['timestamp']).dt.date.min()} a {pd.to_datetime(df['timestamp']).dt.date.max()}")
        
        # Verificar integridad del dataset
        print(f"\n🔍 VERIFICACIÓN DE INTEGRIDAD:")
        print(f"   Columnas disponibles: {len(df.columns)}")
        print(f"   Registros totales: {len(df):,}")
        print(f"   Dispositivos únicos: {df['device_id'].nunique()}")
        
        # Análisis de la variable objetivo (ya incluida)
        anomaly_distribution = df['is_anomaly'].value_counts()
        print(f"   Casos normales: {anomaly_distribution[0]:,} ({anomaly_distribution[0]/len(df):.1%})")
        print(f"   Casos anómalos: {anomaly_distribution[1]:,} ({anomaly_distribution[1]/len(df):.1%})")
        
        # Verificar disponibilidad de características requeridas
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            print(f"⚠️ Características faltantes: {missing_features}")
            # Crear características faltantes si es necesario
            if 'minute' not in df.columns:
                df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
                print("✅ Característica 'minute' creada")
        
        # Preparar datos para entrenamiento
        print(f"\n⚙️ PREPARACIÓN DE CARACTERÍSTICAS:")
        print(f"   Características seleccionadas: {len(self.feature_columns)}")
        for i, feature in enumerate(self.feature_columns, 1):
            print(f"   {i:2d}. {feature}")
        
        # Extraer características y variable objetivo
        X = df[self.feature_columns]  # Características expandidas
        y = df['is_anomaly']         # Variable objetivo ya calculada
        
        # Verificar y manejar valores faltantes
        missing_values = X.isnull().sum()
        if missing_values.any():
            print(f"\n⚠️ VALORES FALTANTES:")
            for col, missing in missing_values[missing_values > 0].items():
                print(f"   {col}: {missing} valores")
            X = X.fillna(X.mean())  # Rellenar con media
            print("✅ Valores faltantes rellenados")
        
        # Normalización de características
        X_scaled = self.scaler.fit_transform(X)
        print(f"📏 Características normalizadas: {X_scaled.shape}")
        
        # División estratificada para mantener proporción de clases
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.3,      # 30% para evaluación
            random_state=42,    # Reproducibilidad
            stratify=y          # Mantener proporción de anomalías
        )
        
        print(f"\n📚 DIVISIÓN DE DATOS:")
        print(f"   Entrenamiento: {len(X_train):,} muestras")
        print(f"   Evaluación: {len(X_test):,} muestras")
        print(f"   Anomalías entrenamiento: {y_train.sum():,} ({y_train.mean():.1%})")
        print(f"   Anomalías evaluación: {y_test.sum():,} ({y_test.mean():.1%})")
        
        # ENTRENAMIENTO MODELO 1: Random Forest (Supervisado)
        print(f"\n🌳 ENTRENANDO RANDOM FOREST (Supervisado)...")
        self.model = RandomForestClassifier(
            n_estimators=150,        # Más árboles para mayor precisión
            random_state=42,         # Reproducibilidad
            max_depth=15,           # Profundidad mayor para dataset complejo
            min_samples_split=10,    # Evitar overfitting
            min_samples_leaf=5,      # Mínimo en hojas
            class_weight='balanced'  # Balance para clases desbalanceadas
        )
        self.model.fit(X_train, y_train)
        
        # Evaluar Random Forest
        rf_accuracy = self.model.score(X_test, y_test)
        rf_predictions = self.model.predict(X_test)
        rf_precision = np.mean((rf_predictions == 1) & (y_test == 1)) / np.max([np.mean(rf_predictions == 1), 0.001])
        rf_recall = np.mean((rf_predictions == 1) & (y_test == 1)) / np.max([np.mean(y_test == 1), 0.001])
        
        print(f"   ✅ Accuracy: {rf_accuracy:.4f}")
        print(f"   ✅ Precision: {rf_precision:.4f}")
        print(f"   ✅ Recall: {rf_recall:.4f}")
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 TOP 5 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        # ENTRENAMIENTO MODELO 2: Isolation Forest (No supervisado)
        print(f"\n🔍 ENTRENANDO ISOLATION FOREST (No supervisado)...")
        self.anomaly_detector = IsolationForest(
            contamination=y_train.mean(),  # Usar tasa real de anomalías
            random_state=42,               # Reproducibilidad
            n_estimators=150,              # Más estimadores
            max_samples='auto',            # Muestras automáticas
            bootstrap=True                 # Bootstrap para robustez
        )
        self.anomaly_detector.fit(X_train)
        
        # Evaluar Isolation Forest
        if_predictions = self.anomaly_detector.predict(X_test)
        if_predictions = np.where(if_predictions == -1, 1, 0)  # Convertir -1 a 1
        if_accuracy = np.mean(if_predictions == y_test)
        
        print(f"   ✅ Accuracy: {if_accuracy:.4f}")
        
        # EVALUACIÓN COMBINADA
        print(f"\n📊 RENDIMIENTO COMBINADO:")
        
        # Consenso entre modelos
        consensus_predictions = (rf_predictions + if_predictions) >= 1
        consensus_accuracy = np.mean(consensus_predictions == y_test)
        
        print(f"   🤖 Random Forest solo: {rf_accuracy:.4f}")
        print(f"   🔍 Isolation Forest solo: {if_accuracy:.4f}")
        print(f"   🎯 Consenso (OR): {consensus_accuracy:.4f}")
        
        # Marcar como entrenado y guardar
        self.is_trained = True
        self.save_models()
        
        print(f"\n✅ ENTRENAMIENTO COMPLETADO:")
        print(f"   📈 Modelos entrenados con {len(X_train):,} muestras")
        print(f"   🎯 {len(self.feature_columns)} características utilizadas")
        print(f"   💾 Modelos guardados para uso futuro")
        print(f"   🚀 Sistema listo para detección en tiempo real")
        
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
        Guarda los modelos entrenados junto con las características utilizadas.
        
        ACTUALIZADO: Incluye las características expandidas y metadatos del dataset sintético.
        """
        model_data = {
            'classifier': self.model,
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,  # NUEVO: Guardar características
            'is_trained': self.is_trained,
            'dataset_version': 'synthetic_7days',      # NUEVO: Versión del dataset
            'n_features': len(self.feature_columns),   # NUEVO: Número de características
            'threshold_basic': self.threshold_basic
        }
        
        with open('ml_models_7days.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("💾 Modelos del dataset 7 días guardados en 'ml_models_7days.pkl'")
        
    def load_models(self):
        """
        Carga modelos pre-entrenados con verificación de compatibilidad.
        
        ACTUALIZADO: Verifica compatibilidad con características expandidas.
        """
        try:
            # Intentar cargar modelos del dataset 7 días primero
            if os.path.exists('ml_models_7days.pkl'):
                with open('ml_models_7days.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.model = model_data['classifier']
                self.anomaly_detector = model_data['anomaly_detector']
                self.scaler = model_data['scaler']
                
                # Verificar compatibilidad de características
                if 'feature_columns' in model_data:
                    loaded_features = model_data['feature_columns']
                    if loaded_features != self.feature_columns:
                        print("⚠️ Características del modelo difieren de las actuales")
                        print(f"   Modelo: {len(loaded_features)} características")
                        print(f"   Actual: {len(self.feature_columns)} características")
                        # Usar características del modelo cargado
                        self.feature_columns = loaded_features
                        
                self.is_trained = True
                print("✅ Modelos del dataset 7 días cargados exitosamente")
                print(f"   📊 Características: {len(self.feature_columns)}")
                return True
                
            # Fallback a modelos antiguos si existen
            elif os.path.exists('ml_models.pkl'):
                print("⚠️ Cargando modelos antiguos (compatibilidad limitada)")
                with open('ml_models.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.model = model_data['classifier']
                self.anomaly_detector = model_data['anomaly_detector']
                self.scaler = model_data['scaler']
                
                # Usar características básicas para compatibilidad
                self.feature_columns = ['humidity_pct', 'raw_value', 'hour', 'minute']
                self.is_trained = True
                print("✅ Modelos antiguos cargados (funcionalidad limitada)")
                return True
                
            else:
                print("⚠️ No se encontraron modelos pre-entrenados")
                return False
                
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            return False
            
    def predict_anomaly(self, raw_value, humidity_pct, hour=None, minute=None, 
                       day_of_week=None, is_weekend=None, is_night=None):
        """
        Predice si una lectura del sensor indica anomalía usando ML inteligente.
        
        ACTUALIZADO: Aprovecha las nuevas características del dataset sintético
        para predicciones más precisas y contextualizadas.
        
        Este es el corazón del sistema de detección en tiempo real. Combina
        múltiples enfoques de ML con características temporales y contextuales
        avanzadas para una detección más robusta y confiable.
        
        Proceso de predicción mejorado:
        1. Verificar disponibilidad de modelos entrenados
        2. Calcular características contextuales automáticamente
        3. Estimar características avanzadas (riesgo, estabilidad, cambios)
        4. Ejecutar predicción con Random Forest (supervisado)
        5. Ejecutar detección con Isolation Forest (no supervisado)
        6. Combinar resultados con lógica de consenso mejorada
        7. Calcular nivel de confianza contextualizado
        
        Args:
            raw_value (int): Valor crudo del sensor de humedad
            humidity_pct (float): Porcentaje de humedad calculado
            hour (int, optional): Hora actual (0-23)
            minute (int, optional): Minuto actual (0-59)
            day_of_week (int, optional): Día de semana (0-6)
            is_weekend (bool, optional): ¿Es fin de semana?
            is_night (bool, optional): ¿Es horario nocturno?
            
        Returns:
            tuple: (is_anomaly, method, confidence, anomaly_score, context_info)
            - is_anomaly (bool): ¿Se detectó anomalía?
            - method (str): Método de detección utilizado
            - confidence (float): Nivel de confianza (0.0-1.0)
            - anomaly_score (float): Score numérico de anomalía
            - context_info (dict): Información contextual de la predicción
        """
        # Verificar disponibilidad de modelos
        if not self.is_trained:
            print("⚠️ Modelos no entrenados, intentando cargar...")
            if not self.load_models():
                print("❌ Modelos no disponibles, usando detección básica")
                is_basic_anomaly = humidity_pct > self.threshold_basic
                context_info = {
                    'method': 'basic_threshold',
                    'threshold': self.threshold_basic,
                    'features_used': 1
                }
                return is_basic_anomaly, "Umbral básico (50%)", 0.6, 0.0, context_info
        
        # Calcular características contextuales automáticamente
        now = datetime.now()
        
        # Características temporales
        if hour is None:
            hour = now.hour
        if minute is None:
            minute = now.minute
        if day_of_week is None:
            day_of_week = now.weekday()  # 0=Lunes, 6=Domingo
        if is_weekend is None:
            is_weekend = 1 if day_of_week >= 5 else 0  # Sábado=5, Domingo=6
        if is_night is None:
            is_night = 1 if hour < 6 or hour > 22 else 0  # 22:00 - 06:00
        
        # Estimar características avanzadas basadas en valores actuales
        # (En un sistema real, estas vendrían de cálculos históricos)
        
        # 1. Normalizar valor raw (estimación basada en rango típico 0-1024)
        raw_normalized = min(max(raw_value / 1024.0, 0), 1)
        
        # 2. Categoría de humedad basada en rangos estándar
        if humidity_pct < 25:
            humidity_category = 0  # Baja
        elif humidity_pct < 60:
            humidity_category = 1  # Media
        else:
            humidity_category = 2  # Alta
        
        # 3. Nivel de riesgo de humedad (función escalada)
        if humidity_pct < 20:
            humidity_risk_level = 0.1
        elif humidity_pct < 40:
            humidity_risk_level = 0.3
        elif humidity_pct < 60:
            humidity_risk_level = 0.6
        else:
            humidity_risk_level = 0.8
        
        # 4. Estabilidad del sensor (simulada como alta por defecto)
        sensor_stability = 1.0  # En sistema real, se calcularía de lecturas recientes
        
        # 5. Cambios en humedad y raw (estimados como promedio para nueva lectura)
        humidity_change = 2.5  # Promedio típico del dataset
        raw_change = 10.0      # Promedio típico del dataset
        
        # Crear vector de características completo
        features = np.array([[
            humidity_pct,           # 0
            raw_value,             # 1  
            raw_normalized,        # 2
            hour,                  # 3
            day_of_week,          # 4
            is_weekend,           # 5
            is_night,             # 6
            humidity_category,     # 7
            humidity_risk_level,   # 8
            sensor_stability,      # 9
            humidity_change,       # 10
            raw_change            # 11
        ]])
        
        # Aplicar normalización entrenada
        features_scaled = self.scaler.transform(features)
        
        # ============= PREDICCIÓN CON MODELO SUPERVISADO =============
        # Random Forest da probabilidades de clase
        prob_anomaly = self.model.predict_proba(features_scaled)[0][1]  # Probabilidad de anomalía
        is_anomaly_classifier = self.model.predict(features_scaled)[0]   # Predicción binaria
        
        # ============= DETECCIÓN CON MODELO NO SUPERVISADO =============
        # Isolation Forest da score de anomalía y predicción binaria
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly_detector = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # ============= LÓGICA DE CONSENSO INTELIGENTE MEJORADA =============
        confidence = prob_anomaly  # Confianza base del clasificador
        is_anomaly = False
        method = "ML Consenso Avanzado"
        
        # Ajustes contextuales de confianza
        confidence_boost = 0.0
        
        # Boost por contexto temporal riesgoso
        if is_night:
            confidence_boost += 0.05  # Noches más problemáticas
        if is_weekend:
            confidence_boost += 0.03  # Fines de semana menos monitoreados
        
        # Boost por niveles de riesgo altos
        if humidity_risk_level > 0.6:
            confidence_boost += 0.1
        
        # Boost por estabilidad baja del sensor
        if sensor_stability < 0.8:
            confidence_boost += 0.05
        
        if is_anomaly_classifier and is_anomaly_detector:
            # Ambos detectan anomalía → MÁXIMA CONFIANZA
            is_anomaly = True
            method = "ML Alto Riesgo (Consenso Doble)"
            confidence = min(prob_anomaly + 0.2 + confidence_boost, 1.0)
            
        elif is_anomaly_classifier:
            # Solo supervisado detecta → CONFIANZA MEDIA-ALTA
            is_anomaly = True
            method = "ML Clasificador (RF)"
            confidence = min(prob_anomaly + confidence_boost, 1.0)
            
        elif is_anomaly_detector:
            # Solo no supervisado detecta → CONFIANZA MEDIA
            is_anomaly = True
            method = "ML Detector Anomalías (IF)"
            confidence = min(0.7 + confidence_boost, 1.0)
            
        else:
            # Ninguno detecta → NORMAL
            is_anomaly = False
            method = "ML Normal"
            confidence = max(1.0 - prob_anomaly - confidence_boost/2, 0.0)
        
        # Información contextual para debugging y análisis
        context_info = {
            'features_used': len(self.feature_columns),
            'temporal_context': {
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': bool(is_weekend),
                'is_night': bool(is_night)
            },
            'risk_context': {
                'humidity_category': humidity_category,
                'humidity_risk_level': humidity_risk_level,
                'sensor_stability': sensor_stability
            },
            'ml_scores': {
                'rf_probability': prob_anomaly,
                'if_score': anomaly_score,
                'confidence_boost': confidence_boost
            },
            'predictions': {
                'rf_prediction': bool(is_anomaly_classifier),
                'if_prediction': bool(is_anomaly_detector)
            }
        }
        
        return is_anomaly, method, confidence, anomaly_score, context_info
    
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
