# ml_analysis.py
"""
Análisis de Machine Learning para DryWall Alert
Grupo 3 - Detección de anomalías y clasificación en datos de sensores de humedad

Este archivo implementa un sistema completo de análisis de Machine Learning que:
1. Carga y prepara datos de sensores de humedad
2. Implementa 10+ algoritmos de ML para detectar filtraciones
3. Compara el rendimiento de todos los modelos
4. Genera visualizaciones y reportes automáticos

El objetivo es encontrar el mejor modelo para detectar anomalías en tiempo real
y integrarlo con el sistema de alertas de WhatsApp.
"""

# Librerías básicas para manipulación de datos y visualización
import pandas as pd  # Para manejo de dataframes y datos estructurados
import numpy as np   # Para operaciones matemáticas y arrays
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones
import seaborn as sns  # Para visualizaciones estadísticas avanzadas
from datetime import datetime  # Para manejo de fechas y timestamps
import warnings
warnings.filterwarnings('ignore')  # Suprimir advertencias para output más limpio

# Importación de modelos de Machine Learning de scikit-learn
# Cada modelo tiene un propósito específico en nuestro análisis:

# Modelos para detección de anomalías (datos sin etiquetas):
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# - IsolationForest: Detecta anomalías aislando puntos atípicos en el espacio de características
# - RandomForest: Conjunto de árboles de decisión para clasificación robusta
# - AdaBoost: Mejora modelos débiles combinándolos de forma adaptativa
# - GradientBoosting: Construye modelos de forma secuencial corrigiendo errores previos

from sklearn.svm import OneClassSVM, SVC
# - OneClassSVM: SVM para detección de anomalías con una sola clase
# - SVC: Support Vector Classifier para clasificación binaria/multiclase

from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
# - KNeighborsClassifier: Clasifica basado en los k vecinos más cercanos
# - LocalOutlierFactor: Detecta anomalías comparando densidad local con vecinos

from sklearn.cluster import DBSCAN
# - DBSCAN: Clustering que identifica ruido/anomalías como puntos aislados

from sklearn.neural_network import MLPClassifier
# - MLPClassifier: Red neuronal multicapa para clasificación no lineal

from sklearn.tree import DecisionTreeClassifier
# - DecisionTreeClassifier: Árbol de decisión para clasificación interpretable

# Herramientas para división de datos y validación
from sklearn.model_selection import train_test_split, cross_val_score
# - train_test_split: Divide datos en entrenamiento y prueba
# - cross_val_score: Validación cruzada para evaluar modelos

# Preprocesamiento de datos
from sklearn.preprocessing import StandardScaler, LabelEncoder
# - StandardScaler: Normaliza características para que tengan media 0 y desviación 1
# - LabelEncoder: Codifica etiquetas categóricas como números

# Métricas de evaluación
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
# - accuracy_score: Porcentaje de predicciones correctas
# - f1_score: Media armónica entre precisión y recall
# - precision_score: De las predicciones positivas, cuántas son correctas
# - recall_score: De los casos positivos reales, cuántos detectamos

# Importación opcional de TensorFlow para redes neuronales avanzadas
# TensorFlow se usa para crear un Autoencoder (red neuronal para detección de anomalías)
try:
    import tensorflow as tf  # Framework de deep learning de Google
    from tensorflow.keras.models import Model  # Para definir arquitecturas de red
    from tensorflow.keras.layers import Dense, Input  # Capas densas y de entrada
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow disponible - Se usarán 10 modelos incluyendo Autoencoder")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Se usarán 9 modelos (suficiente para el análisis)")
    print("💡 Para instalar TensorFlow: pip install tensorflow")

class DryWallAnalyzer:
    """
    Clase principal para análisis de Machine Learning en el sistema DryWall Alert.
    
    Esta clase implementa un pipeline completo de ML que:
    1. Carga datos de sensores de humedad desde CSV
    2. Prepara y transforma los datos para análisis
    3. Ejecuta 10+ algoritmos de ML diferentes
    4. Compara rendimiento y selecciona el mejor modelo
    5. Genera visualizaciones y reportes automáticos
    
    El objetivo es detectar filtraciones/anomalías en tiempo real
    para el sistema de alertas de WhatsApp.
    """
    
    def __init__(self, data_file='humedad_datos.csv'):
        """
        Inicializa el analizador con los datos del sensor de humedad
        
        Args:
            data_file (str): Ruta al archivo CSV con datos del sensor
            
        Atributos:
            data_file: Archivo de datos
            df: DataFrame con datos cargados
            X: Matriz de características (features)
            y: Vector de etiquetas (target)
            X_train, X_test, y_train, y_test: Datos divididos para entrenamiento/prueba
            scaler: Normalizador de datos
            results: Diccionario con resultados de todos los modelos
        """
        self.data_file = data_file
        self.df = None  # DataFrame principal con todos los datos
        self.X = None   # Características (raw, humidity_pct, hour, minute)
        self.y = None   # Variable objetivo (is_anomaly: 0=normal, 1=anomalía)
        
        # Datos divididos para entrenamiento y evaluación
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Herramientas de preprocesamiento y almacenamiento de resultados
        self.scaler = StandardScaler()  # Para normalizar datos
        self.results = {}  # Diccionario para guardar métricas de cada modelo
        
    def load_and_prepare_data(self):
        """
        Carga y prepara los datos para el análisis de Machine Learning.
        
        Este método:
        1. Lee el archivo CSV con datos del sensor
        2. Extrae características temporales (hora, minuto)
        3. Crea variables objetivo para clasificación
        4. Normaliza los datos para los algoritmos ML
        5. Muestra estadísticas descriptivas
        
        Variables creadas:
        - is_anomaly: 1 si humedad > 50% (indica filtración), 0 si normal
        - risk_level: Categorías de riesgo (Bajo/Normal/Alto/Crítico)
        - hour/minute: Características temporales extraídas del timestamp
        """
        print("📊 Cargando datos del sensor de humedad...")
        self.df = pd.read_csv(self.data_file)
        
        # Información básica del dataset para entender los datos
        print(f"✅ Datos cargados: {len(self.df)} registros")
        print(f"📋 Columnas disponibles: {list(self.df.columns)}")
        print(f"📊 Estadísticas básicas de los datos:")
        print(self.df.describe())
        
        # Feature Engineering: Crear características adicionales desde timestamp
        # Convertir timestamp a hora y minuto para capturar patrones temporales
        # Ejemplo: "14:30:25" -> hour=14, minute=30
        self.df['hour'] = pd.to_datetime(self.df['timestamp'], format='%H:%M:%S').dt.hour
        self.df['minute'] = pd.to_datetime(self.df['timestamp'], format='%H:%M:%S').dt.minute
        
        # Crear variable objetivo para clasificación binaria
        # CRITERIO CLAVE: Humedad > 50% indica posible filtración
        # Esto se basa en estándares de construcción donde >50% es problemático
        self.df['is_anomaly'] = (self.df['humidity_pct'] > 50).astype(int)
        
        # Crear clasificación por niveles de riesgo para análisis adicional
        self.df['risk_level'] = pd.cut(
            self.df['humidity_pct'], 
            bins=[0, 20, 40, 60, 100],  # Rangos de humedad
            labels=['Bajo', 'Normal', 'Alto', 'Crítico']  # Etiquetas descriptivas
        )
        
        # Seleccionar características (features) para el modelo
        # Incluimos valor raw del sensor, porcentaje de humedad y tiempo
        feature_columns = ['raw', 'humidity_pct', 'hour', 'minute']
        self.X = self.df[feature_columns]  # Matriz de características
        self.y = self.df['is_anomaly']     # Vector de etiquetas objetivo
        
        # Normalización de características para algoritmos sensibles a escala
        # StandardScaler convierte datos a media=0, desviación=1
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Mostrar distribución de clases para verificar balance
        print(f"\n🎯 Distribución de casos:")
        print(f"   Normal (0): {(self.y == 0).sum()} casos")
        print(f"   Anomalía (1): {(self.y == 1).sum()} casos")
        print(f"   Porcentaje anomalías: {(self.y == 1).mean():.2%}")
        
        print(f"\n🎯 Distribución por nivel de riesgo:")
        print(self.df['risk_level'].value_counts())
        
    def split_data(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Usa estratificación para mantener la proporción de clases en ambos conjuntos.
        Esto es crucial cuando hay desbalance de clases (pocas anomalías vs muchos casos normales).
        
        División típica:
        - 70% para entrenamiento (el modelo aprende patrones)
        - 30% para prueba (evaluamos qué tan bien generaliza)
        
        La semilla random_state=42 asegura resultados reproducibles.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled,  # Datos normalizados
            self.y,         # Etiquetas objetivo
            test_size=0.3,  # 30% para prueba
            random_state=42,  # Semilla para reproducibilidad
            stratify=self.y   # Mantener proporción de clases
        )
        
        print(f"📊 Datos de entrenamiento: {len(self.X_train)} muestras")
        print(f"📊 Datos de prueba: {len(self.X_test)} muestras")
        print(f"📊 Proporción entrenamiento/prueba: {len(self.X_train)/len(self.X_test):.1f}:1")
        
    def visualize_data(self):
        """
        Crea visualizaciones exploratorias para entender los datos.
        
        Genera 6 gráficos diferentes:
        1. Histograma de distribución de humedad
        2. Scatter plot raw vs humedad coloreado por anomalías
        3. Serie temporal de humedad
        4. Promedio de humedad por hora del día
        5. Boxplot de valores raw por nivel de riesgo
        6. Matriz de correlación entre variables
        
        Estas visualizaciones ayudan a:
        - Identificar patrones temporales
        - Detectar correlaciones entre variables
        - Entender la distribución de anomalías
        - Validar la calidad de los datos
        """
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Distribución de humedad - ¿Los datos están balanceados?
        plt.subplot(2, 3, 1)
        plt.hist(self.df['humidity_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribución de Humedad %\n(¿Hay sesgo hacia ciertos valores?)')
        plt.xlabel('Humedad %')
        plt.ylabel('Frecuencia')
        plt.axvline(x=50, color='red', linestyle='--', label='Umbral anomalía (50%)')
        plt.legend()
        
        # Gráfico 2: Relación raw vs humedad con anomalías
        plt.subplot(2, 3, 2)
        scatter = plt.scatter(self.df['raw'], self.df['humidity_pct'], 
                            c=self.df['is_anomaly'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Anomalía (0=Normal, 1=Anomalía)')
        plt.title('Sensor Raw vs Humedad %\n(Coloreado por anomalías)')
        plt.xlabel('Valor Raw del Sensor')
        plt.ylabel('Humedad %')
        
        # Gráfico 3: Serie temporal - ¿Hay tendencias o estacionalidad?
        plt.subplot(2, 3, 3)
        plt.plot(self.df.index, self.df['humidity_pct'], alpha=0.7, linewidth=1)
        plt.title('Serie Temporal de Humedad\n(¿Hay patrones temporales?)')
        plt.xlabel('Tiempo (índice de muestra)')
        plt.ylabel('Humedad %')
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Umbral crítico')
        plt.legend()
        
        # Gráfico 4: Patrones por hora del día
        plt.subplot(2, 3, 4)
        hourly_avg = self.df.groupby('hour')['humidity_pct'].mean()
        plt.bar(hourly_avg.index, hourly_avg.values, alpha=0.7, color='orange')
        plt.title('Humedad Promedio por Hora\n(¿Hay horas más problemáticas?)')
        plt.xlabel('Hora del día')
        plt.ylabel('Humedad promedio %')
        plt.xticks(range(0, 24, 2))  # Mostrar cada 2 horas
        
        # Gráfico 5: Distribución de valores raw por nivel de riesgo
        plt.subplot(2, 3, 5)
        sns.boxplot(data=self.df, x='risk_level', y='raw')
        plt.title('Distribución de Valores Raw\npor Nivel de Riesgo')
        plt.xlabel('Nivel de Riesgo')
        plt.ylabel('Valor Raw del Sensor')
        plt.xticks(rotation=45)
        
        # Gráfico 6: Matriz de correlación - ¿Qué variables están relacionadas?
        plt.subplot(2, 3, 6)
        correlation_matrix = self.df[['raw', 'humidity_pct', 'hour', 'minute', 'is_anomaly']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True)
        plt.title('Matriz de Correlación\n(Relaciones entre variables)')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualizaciones guardadas en 'data_analysis.png'")
        plt.show()
        
    def evaluate_model(self, model, X_test, y_test, model_name, is_supervised=True):
        """
        Evalúa el rendimiento de un modelo de ML y guarda las métricas.
        
        Esta función es el corazón de la evaluación de modelos. Calcula métricas
        estándar para clasificación binaria y maneja tanto modelos supervisados
        como no supervisados (detección de anomalías).
        
        Args:
            model: Modelo entrenado de ML
            X_test: Datos de prueba (características)
            y_test: Etiquetas verdaderas de prueba
            model_name: Nombre del modelo para mostrar resultados
            is_supervised: Si es modelo supervisado (usa etiquetas) o no supervisado
            
        Métricas calculadas:
        - Accuracy: % de predicciones correctas (TP+TN)/(TP+TN+FP+FN)
        - F1-Score: Media armónica de precisión y recall: 2*(P*R)/(P+R)
        - Precision: De las predicciones positivas, % correctas: TP/(TP+FP)
        - Recall: De los casos positivos reales, % detectados: TP/(TP+FN)
        
        Para nuestro problema:
        - TP (True Positive): Filtraciones detectadas correctamente
        - FP (False Positive): Falsas alarmas
        - TN (True Negative): Casos normales identificados correctamente
        - FN (False Negative): Filtraciones NO detectadas (¡MUY PELIGROSO!)
        """
        try:
            if is_supervised:
                # Modelos supervisados: usan etiquetas para entrenamiento
                y_pred = model.predict(X_test)
                
                # Calcular métricas de clasificación binaria
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                # Guardar resultados en diccionario para comparación posterior
                self.results[model_name] = {
                    'accuracy': accuracy,      # Exactitud general
                    'f1_score': f1,           # Balance precisión-recall
                    'precision': precision,    # Calidad de predicciones positivas
                    'recall': recall,         # Capacidad de detectar casos positivos
                    'predictions': y_pred     # Predicciones para análisis adicional
                }
                
                print(f"\n📊 {model_name}:")
                print(f"   Accuracy: {accuracy:.4f} ({'Excelente' if accuracy > 0.9 else 'Bueno' if accuracy > 0.8 else 'Regular'})")
                print(f"   F1-Score: {f1:.4f} (Balance precisión-recall)")
                print(f"   Precision: {precision:.4f} (Calidad predicciones)")
                print(f"   Recall: {recall:.4f} (Detección anomalías)")
                
            else:
                # Modelos no supervisados: detectan anomalías sin etiquetas
                y_pred = model.fit_predict(X_test)
                
                # Algunos modelos retornan -1 para anomalías, convertir a 1
                if hasattr(model, 'predict') and model_name in ['Isolation Forest', 'One-Class SVM', 'LOF']:
                    y_pred = np.where(y_pred == -1, 1, 0)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'type': 'unsupervised'
                }
                
                print(f"\n📊 {model_name} (Detección no supervisada):")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                print(f"   📝 Nota: Modelo aprende patrones sin etiquetas previas")
                
        except Exception as e:
            print(f"❌ Error evaluando {model_name}: {e}")
            print(f"💡 Posibles causas: Datos insuficientes, parámetros incorrectos, incompatibilidad")
            self.results[model_name] = {'error': str(e)}
    
    def build_autoencoder(self, input_dim):
        """
        Construye una red neuronal Autoencoder para detección de anomalías.
        
        Un Autoencoder es una red neuronal que aprende a reconstruir sus entradas.
        Para detección de anomalías funciona así:
        
        1. Se entrena solo con datos "normales"
        2. Aprende a comprimir y reconstruir estos datos
        3. Datos anómalos tendrán mayor error de reconstrucción
        4. Si error > umbral → anomalía detectada
        
        Arquitectura:
        - Encoder: 4 features → 8 → 4 → 2 (compresión)
        - Decoder: 2 → 4 → 8 → 4 features (reconstrucción)
        
        Args:
            input_dim: Número de características de entrada (4 en nuestro caso)
            
        Returns:
            Modelo de autoencoder compilado y listo para entrenar
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow no disponible, no se puede crear Autoencoder")
            return None
            
        print("🧠 Construyendo Autoencoder para detección de anomalías...")
        
        # Capa de entrada
        input_layer = Input(shape=(input_dim,))
        
        # Encoder: Reduce dimensionalidad progresivamente
        encoder = Dense(8, activation='relu', name='encoder_layer1')(input_layer)
        encoder = Dense(4, activation='relu', name='encoder_layer2')(encoder)
        encoder = Dense(2, activation='relu', name='encoder_bottleneck')(encoder)  # Cuello de botella
        
        # Decoder: Reconstruye dimensionalidad original
        decoder = Dense(4, activation='relu', name='decoder_layer1')(encoder)
        decoder = Dense(8, activation='relu', name='decoder_layer2')(decoder)
        decoder = Dense(input_dim, activation='linear', name='decoder_output')(decoder)  # Sin activación final
        
        # Crear modelo completo
        autoencoder = Model(input_layer, decoder, name='DryWall_Autoencoder')
        
        # Compilar con optimizador Adam y función de pérdida MSE
        autoencoder.compile(
            optimizer='adam',        # Optimizador adaptativo eficiente
            loss='mse',             # Error cuadrático medio para reconstrucción
            metrics=['mae']         # Error absoluto medio como métrica adicional
        )
        
        print("✅ Autoencoder construido - Arquitectura: 4→8→4→2→4→8→4")
        return autoencoder
    
    def run_all_models(self):
        """
        Ejecuta todos los algoritmos de Machine Learning requeridos.
        
        Esta función implementa 10 modelos diferentes divididos en categorías:
        
        🔍 DETECCIÓN DE ANOMALÍAS (No supervisados - aprenden sin etiquetas):
        1. Isolation Forest: Aísla puntos atípicos en espacios de menor dimensión
        2. One-Class SVM: Define una frontera alrededor de datos "normales"
        3. Autoencoder: Red neuronal que detecta por error de reconstrucción
        4. DBSCAN: Clustering que identifica ruido como anomalías
        5. LOF: Compara densidad local de cada punto con sus vecinos
        
        � CLASIFICACIÓN SUPERVISADA (Aprenden con ejemplos etiquetados):
        6. Random Forest: Ensemble de árboles de decisión
        7. k-NN: Clasifica por vecinos más cercanos
        8. MLP: Red neuronal multicapa
        9. AdaBoost: Boosting adaptativo
        10. Gradient Boosting: Boosting por gradiente
        11. SVM: Máquinas de vectores de soporte
        
        Cada modelo se evalúa con las mismas métricas para comparación justa.
        """
        print("\n🚀 Iniciando análisis comparativo con múltiples algoritmos de ML...")
        print("📋 Modelos a evaluar:")
        print("   🔍 Detección de anomalías: Isolation Forest, One-Class SVM, DBSCAN, LOF")
        print("   📊 Clasificación supervisada: Random Forest, k-NN, MLP, AdaBoost, GB")
        print("   🧠 Deep Learning: Autoencoder (si TensorFlow disponible)")
        
        # ==================== MODELOS DE DETECCIÓN DE ANOMALÍAS ====================
        
        # 1. ISOLATION FOREST
        # Principio: Aísla anomalías construyendo árboles aleatorios
        # Las anomalías requieren menos divisiones para ser aisladas
        print("\n1️⃣ Isolation Forest - Detección por aislamiento")
        iso_forest = IsolationForest(
            contamination=0.1,    # Esperamos ~10% de anomalías
            random_state=42,      # Reproducibilidad
            n_estimators=100      # Número de árboles
        )
        iso_forest.fit(self.X_train)  # Entrena solo con datos de entrenamiento
        y_pred_iso = iso_forest.predict(self.X_test)
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convertir -1→1 (anomalía), 1→0 (normal)
        
        accuracy = accuracy_score(self.y_test, y_pred_iso)
        f1 = f1_score(self.y_test, y_pred_iso, average='weighted')
        self.results['Isolation Forest'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_iso,
            'description': 'Detecta anomalías por facilidad de aislamiento'
        }
        print(f"   📊 Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 2. ONE-CLASS SVM
        # Principio: Aprende una frontera que encierra datos "normales"
        # Puntos fuera de la frontera son anomalías
        print("\n2️⃣ One-Class SVM - Frontera de normalidad")
        oc_svm = OneClassSVM(
            gamma='scale',  # Parámetro del kernel RBF
            nu=0.1         # Fracción esperada de anomalías
        )
        oc_svm.fit(self.X_train)
        y_pred_svm = oc_svm.predict(self.X_test)
        y_pred_svm = np.where(y_pred_svm == -1, 1, 0)
        
        accuracy = accuracy_score(self.y_test, y_pred_svm)
        f1 = f1_score(self.y_test, y_pred_svm, average='weighted')
        self.results['One-Class SVM'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_svm,
            'description': 'Define frontera de datos normales'
        }
        print(f"   📊 Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 3. AUTOENCODER (si TensorFlow disponible)
        # Principio: Red neuronal que reconstruye entradas
        # Mayor error de reconstrucción = anomalía
        if TENSORFLOW_AVAILABLE:
            print("\n3️⃣ Autoencoder - Detección por error de reconstrucción")
            autoencoder = self.build_autoencoder(self.X_train.shape[1])
            
            # Entrenar solo con datos normales para mejor detección
            normal_indices = self.y_train == 0
            X_train_normal = self.X_train[normal_indices]
            
            # Entrenamiento silencioso
            history = autoencoder.fit(
                X_train_normal, X_train_normal,  # Autoencoder aprende a reconstruir
                epochs=50,          # Número de pasadas por los datos
                batch_size=16,      # Muestras por actualización
                verbose=0,          # Sin output de entrenamiento
                validation_split=0.2  # 20% para validación
            )
            
            # Calcular error de reconstrucción en datos de prueba
            X_test_pred = autoencoder.predict(self.X_test, verbose=0)
            mse = np.mean(np.power(self.X_test - X_test_pred, 2), axis=1)
            
            # Umbral: percentil 90 de errores = top 10% como anomalías
            threshold = np.percentile(mse, 90)
            y_pred_ae = (mse > threshold).astype(int)
            
            accuracy = accuracy_score(self.y_test, y_pred_ae)
            f1 = f1_score(self.y_test, y_pred_ae, average='weighted')
            self.results['Autoencoder'] = {
                'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_ae,
                'description': 'Red neuronal - error de reconstrucción',
                'threshold': threshold, 'mse_scores': mse
            }
            print(f"   📊 Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            print(f"   🎯 Umbral error: {threshold:.6f}")
        else:
            print("\n3️⃣ Autoencoder - ⚠️ Saltado (TensorFlow no disponible)")
            print("   💡 Instalar con: pip install tensorflow")
        
        # ==================== MODELOS DE CLASIFICACIÓN SUPERVISADA ====================
        
        # 4. RANDOM FOREST
        # Principio: Ensemble de árboles de decisión con votación mayoritaria
        # Robusto contra overfitting y maneja bien datos mixtos
        print("\n4️⃣ Random Forest - Ensemble de árboles")
        rf = RandomForestClassifier(
            n_estimators=100,    # Número de árboles
            random_state=42,     # Reproducibilidad
            max_depth=10,        # Profundidad máxima para evitar overfitting
            min_samples_split=5  # Mínimo de muestras para dividir nodo
        )
        rf.fit(self.X_train, self.y_train)
        self.evaluate_model(rf, self.X_test, self.y_test, 'Random Forest')
        
        # 5. k-NEAREST NEIGHBORS
        # Principio: Clasifica basado en las k etiquetas de vecinos más cercanos
        # Simple pero efectivo, especialmente con buenos datos
        print("\n5️⃣ k-Nearest Neighbors - Clasificación por vecindad")
        knn = KNeighborsClassifier(
            n_neighbors=5,      # Usar 5 vecinos más cercanos
            weights='distance'  # Pesar por distancia (vecinos más cerca = mayor peso)
        )
        knn.fit(self.X_train, self.y_train)
        self.evaluate_model(knn, self.X_test, self.y_test, 'k-NN')
        
        # 6. DBSCAN - CLUSTERING
        # Principio: Agrupa puntos densos, marca puntos aislados como ruido
        # Ruido = anomalías en nuestro contexto
        print("\n6️⃣ DBSCAN - Clustering con detección de ruido")
        dbscan = DBSCAN(
            eps=0.5,           # Radio de vecindad
            min_samples=5      # Mínimo de puntos para formar cluster
        )
        clusters = dbscan.fit_predict(self.X_test)
        
        # Puntos de ruido (cluster -1) se consideran anomalías
        y_pred_dbscan = (clusters == -1).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred_dbscan)
        f1 = f1_score(self.y_test, y_pred_dbscan, average='weighted')
        self.results['DBSCAN'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_dbscan,
            'description': 'Clustering - ruido como anomalías',
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'n_noise': list(clusters).count(-1)
        }
        print(f"   📊 Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print(f"   🔍 Clusters encontrados: {self.results['DBSCAN']['n_clusters']}")
        print(f"   🔍 Puntos de ruido: {self.results['DBSCAN']['n_noise']}")
        
        # 7. LOCAL OUTLIER FACTOR
        # Principio: Compara densidad local de cada punto con sus vecinos
        # Puntos en regiones menos densas = anomalías
        print("\n7️⃣ Local Outlier Factor - Densidad local")
        lof = LocalOutlierFactor(
            n_neighbors=20,      # Número de vecinos para calcular densidad
            contamination=0.1    # Fracción esperada de anomalías
        )
        y_pred_lof = lof.fit_predict(self.X_test)
        y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
        
        accuracy = accuracy_score(self.y_test, y_pred_lof)
        f1 = f1_score(self.y_test, y_pred_lof, average='weighted')
        self.results['LOF'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_lof,
            'description': 'Detección por densidad local anómala'
        }
        print(f"   📊 Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 8. MULTI-LAYER PERCEPTRON
        # Principio: Red neuronal con capas ocultas para patrones no lineales
        # Puede aprender relaciones complejas entre características
        print("\n8️⃣ Multi-Layer Perceptron - Red neuronal")
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Dos capas ocultas: 100 y 50 neuronas
            max_iter=500,                  # Máximo de iteraciones
            random_state=42,               # Reproducibilidad
            alpha=0.01                     # Regularización para evitar overfitting
        )
        mlp.fit(self.X_train, self.y_train)
        self.evaluate_model(mlp, self.X_test, self.y_test, 'MLP')
        
        # 9. ADABOOST
        # Principio: Combina modelos débiles, enfocándose en errores previos
        # Cada modelo siguiente corrige errores del anterior
        print("\n9️⃣ AdaBoost - Boosting adaptativo")
        ada = AdaBoostClassifier(
            n_estimators=100,    # Número de estimadores débiles
            random_state=42,     # Reproducibilidad
            learning_rate=1.0    # Contribución de cada estimador
        )
        ada.fit(self.X_train, self.y_train)
        self.evaluate_model(ada, self.X_test, self.y_test, 'AdaBoost')
        
        # 10. GRADIENT BOOSTING
        # Principio: Construye modelos secuencialmente minimizando función de pérdida
        # Muy efectivo pero puede ser propenso a overfitting
        print("\n🔟 Gradient Boosting - Boosting por gradiente")
        gb = GradientBoostingClassifier(
            n_estimators=100,     # Número de etapas de boosting
            random_state=42,      # Reproducibilidad
            learning_rate=0.1,    # Reduce contribución de cada árbol
            max_depth=3           # Profundidad máxima de árboles
        )
        gb.fit(self.X_train, self.y_train)
        self.evaluate_model(gb, self.X_test, self.y_test, 'Gradient Boosting')
        
        # 11. SUPPORT VECTOR MACHINE (Reemplazo si no hay TensorFlow)
        # Principio: Encuentra hiperplano óptimo que separa clases
        # Efectivo en espacios de alta dimensión
        if not TENSORFLOW_AVAILABLE:
            print("\n🆕 SVM (Support Vector Machine) - Hiperplano óptimo")
            svm = SVC(
                kernel='rbf',        # Kernel radial para patrones no lineales
                random_state=42,     # Reproducibilidad
                C=1.0,              # Parámetro de regularización
                gamma='scale'       # Parámetro del kernel
            )
            svm.fit(self.X_train, self.y_train)
            self.evaluate_model(svm, self.X_test, self.y_test, 'SVM')
        
        print(f"\n✅ Análisis completado - {len(self.results)} modelos evaluados")
        print("📊 Procediendo a comparación de rendimiento...")
        # Considerar puntos de ruido (-1) como anomalías
        y_pred_dbscan = (clusters == -1).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred_dbscan)
        f1 = f1_score(self.y_test, y_pred_dbscan, average='weighted')
        self.results['DBSCAN'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_dbscan
        }
        print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 7. LOF (Local Outlier Factor)
        print("\n7️⃣ Local Outlier Factor")
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        y_pred_lof = lof.fit_predict(self.X_test)
        y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
        
        accuracy = accuracy_score(self.y_test, y_pred_lof)
        f1 = f1_score(self.y_test, y_pred_lof, average='weighted')
        self.results['LOF'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_lof
        }
        print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 8. MLP (Multi-Layer Perceptron)
        print("\n8️⃣ Multi-Layer Perceptron")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(self.X_train, self.y_train)
        self.evaluate_model(mlp, self.X_test, self.y_test, 'MLP')
        
        # 9. AdaBoost
        print("\n9️⃣ AdaBoost")
        ada = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada.fit(self.X_train, self.y_train)
        self.evaluate_model(ada, self.X_test, self.y_test, 'AdaBoost')
        
        # 10. Gradient Boosting
        print("\n🔟 Gradient Boosting")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.evaluate_model(gb, self.X_test, self.y_test, 'Gradient Boosting')
        
        # 11. SVM (Support Vector Machine) - Reemplazo de Autoencoder
        if not TENSORFLOW_AVAILABLE:
            print("\n🆕 SVM (Support Vector Machine)")
            svm = SVC(kernel='rbf', random_state=42)
            svm.fit(self.X_train, self.y_train)
            self.evaluate_model(svm, self.X_test, self.y_test, 'SVM')
    
    def compare_models(self):
        """
        Compara el rendimiento de todos los modelos
        """
        print("\n📊 COMPARACIÓN DE MODELOS")
        print("=" * 80)
        
        # Crear DataFrame con resultados
        comparison_data = []
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Modelo': model_name,
                    'Accuracy': metrics['accuracy'],
                    'F1-Score': metrics['f1_score'],
                    'Precision': metrics.get('precision', 'N/A'),
                    'Recall': metrics.get('recall', 'N/A')
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Formatear y mostrar la tabla
        print(comparison_df.round(4).to_string(index=False))
        
        # Visualizar comparación
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras para Accuracy
        plt.subplot(2, 2, 1)
        plt.bar(comparison_df['Modelo'], comparison_df['Accuracy'], color='skyblue')
        plt.title('Accuracy por Modelo')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        
        # Gráfico de barras para F1-Score
        plt.subplot(2, 2, 2)
        plt.bar(comparison_df['Modelo'], comparison_df['F1-Score'], color='lightgreen')
        plt.title('F1-Score por Modelo')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('F1-Score')
        
        # Scatter plot Accuracy vs F1-Score
        plt.subplot(2, 2, 3)
        plt.scatter(comparison_df['Accuracy'], comparison_df['F1-Score'], s=100, alpha=0.7)
        for i, txt in enumerate(comparison_df['Modelo']):
            plt.annotate(txt, (comparison_df['Accuracy'].iloc[i], comparison_df['F1-Score'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Accuracy')
        plt.ylabel('F1-Score')
        plt.title('Accuracy vs F1-Score')
        
        # Heatmap de métricas
        plt.subplot(2, 2, 4)
        metrics_for_heatmap = comparison_df.set_index('Modelo')[['Accuracy', 'F1-Score']]
        sns.heatmap(metrics_for_heatmap.T, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Heatmap de Métricas')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Encontrar el mejor modelo
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 MEJOR MODELO: {best_model['Modelo']}")
        print(f"   📊 F1-Score: {best_model['F1-Score']:.4f}")
        print(f"   🎯 Accuracy: {best_model['Accuracy']:.4f}")
        
        return comparison_df
    
    def generate_report(self):
        """
        Genera un reporte completo del análisis
        """
        print("\n📋 REPORTE FINAL - DRYWALL ALERT")
        print("=" * 60)
        print(f"📊 Dataset: {len(self.df)} registros de sensores de humedad")
        print(f"🎯 Problema: Detección de anomalías/filtraciones")
        print(f"⚙️ Features: {list(self.X.columns)}")
        print(f"🏷️ Clases: Normal (0), Anomalía (1)")
        print(f"📈 Distribución: {self.df['is_anomaly'].value_counts().to_dict()}")
        
        print("\n🔍 JUSTIFICACIÓN DEL MEJOR MODELO:")
        best_models = sorted(self.results.items(), 
                           key=lambda x: x[1].get('f1_score', 0) if 'error' not in x[1] else 0, 
                           reverse=True)[:3]
        
        for i, (model_name, metrics) in enumerate(best_models, 1):
            if 'error' not in metrics:
                print(f"{i}. {model_name}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        best_model_name = best_models[0][0]
        print(f"\n✅ RECOMENDACIÓN: Usar {best_model_name}")
        print("💡 RAZÓN: Mejor balance entre precision y recall para detectar filtraciones")
        print("🛠️ APLICACIÓN: Integrar en el sistema de alertas de WhatsApp")
        
def main():
    """
    Función principal para ejecutar el análisis completo
    """
    print("🏠 DRYWALL ALERT - Análisis de Machine Learning")
    print("=" * 50)
    
    # Inicializar analizador
    analyzer = DryWallAnalyzer()
    
    # Cargar y preparar datos
    analyzer.load_and_prepare_data()
    
    # Dividir datos
    analyzer.split_data()
    
    # Visualizar datos
    print("\n📊 Generando visualizaciones...")
    analyzer.visualize_data()
    
    # Ejecutar todos los modelos
    analyzer.run_all_models()
    
    # Comparar modelos
    comparison_df = analyzer.compare_models()
    
    # Generar reporte final
    analyzer.generate_report()
    
    print("\n✅ Análisis completado!")
    print("📁 Archivos generados:")
    print("   - data_analysis.png (análisis exploratorio)")
    print("   - model_comparison.png (comparación de modelos)")
    
    return analyzer, comparison_df

if __name__ == "__main__":
    analyzer, results = main()
