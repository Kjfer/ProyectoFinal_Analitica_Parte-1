# ml_analysis.py
"""
Análisis de Machine Learning para DryWall Alert
Grupo 3 - Detección de anomalías y clasificación en datos de sensores de humedad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modelos de ML
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Para Autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow disponible")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Se usarán 9 modelos (suficiente para el análisis)")

class DryWallAnalyzer:
    def __init__(self, data_file='humedad_datos.csv'):
        """
        Inicializa el analizador con los datos del sensor de humedad
        """
        self.data_file = data_file
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Carga y prepara los datos para el análisis
        """
        print("📊 Cargando datos...")
        self.df = pd.read_csv(self.data_file)
        
        # Información básica del dataset
        print(f"✅ Datos cargados: {len(self.df)} registros")
        print(f"📋 Columnas: {list(self.df.columns)}")
        print(f"📊 Estadísticas básicas:")
        print(self.df.describe())
        
        # Crear features adicionales
        self.df['hour'] = pd.to_datetime(self.df['timestamp'], format='%H:%M:%S').dt.hour
        self.df['minute'] = pd.to_datetime(self.df['timestamp'], format='%H:%M:%S').dt.minute
        
        # Crear variable objetivo (clasificación)
        # Consideramos "anomalía" cuando la humedad es > 50% (filtración)
        self.df['is_anomaly'] = (self.df['humidity_pct'] > 50).astype(int)
        self.df['risk_level'] = pd.cut(self.df['humidity_pct'], 
                                      bins=[0, 20, 40, 60, 100], 
                                      labels=['Bajo', 'Normal', 'Alto', 'Crítico'])
        
        # Features para el modelo
        feature_columns = ['raw', 'humidity_pct', 'hour', 'minute']
        self.X = self.df[feature_columns]
        self.y = self.df['is_anomaly']
        
        # Normalizar features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"🎯 Distribución de clases:")
        print(self.df['is_anomaly'].value_counts())
        print(f"🎯 Distribución de niveles de riesgo:")
        print(self.df['risk_level'].value_counts())
        
    def split_data(self):
        """
        Divide los datos en entrenamiento y prueba
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        print(f"📊 Datos de entrenamiento: {len(self.X_train)}")
        print(f"📊 Datos de prueba: {len(self.X_test)}")
        
    def visualize_data(self):
        """
        Visualiza los datos para entender mejor el problema
        """
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Distribución de humedad
        plt.subplot(2, 3, 1)
        plt.hist(self.df['humidity_pct'], bins=20, alpha=0.7, color='skyblue')
        plt.title('Distribución de Humedad %')
        plt.xlabel('Humedad %')
        plt.ylabel('Frecuencia')
        
        # Gráfico 2: Raw vs Humedad
        plt.subplot(2, 3, 2)
        scatter = plt.scatter(self.df['raw'], self.df['humidity_pct'], 
                            c=self.df['is_anomaly'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Anomalía')
        plt.title('Raw vs Humedad (coloreado por anomalías)')
        plt.xlabel('Valor Raw')
        plt.ylabel('Humedad %')
        
        # Gráfico 3: Serie temporal
        plt.subplot(2, 3, 3)
        plt.plot(self.df.index, self.df['humidity_pct'], alpha=0.7)
        plt.title('Serie Temporal de Humedad')
        plt.xlabel('Tiempo (índice)')
        plt.ylabel('Humedad %')
        
        # Gráfico 4: Distribución por hora
        plt.subplot(2, 3, 4)
        self.df.groupby('hour')['humidity_pct'].mean().plot(kind='bar')
        plt.title('Humedad Promedio por Hora')
        plt.xlabel('Hora')
        plt.ylabel('Humedad %')
        
        # Gráfico 5: Boxplot por nivel de riesgo
        plt.subplot(2, 3, 5)
        sns.boxplot(data=self.df, x='risk_level', y='raw')
        plt.title('Distribución Raw por Nivel de Riesgo')
        plt.xticks(rotation=45)
        
        # Gráfico 6: Correlación
        plt.subplot(2, 3, 6)
        correlation_matrix = self.df[['raw', 'humidity_pct', 'hour', 'minute', 'is_anomaly']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_model(self, model, X_test, y_test, model_name, is_supervised=True):
        """
        Evalúa un modelo y guarda las métricas
        """
        try:
            if is_supervised:
                y_pred = model.predict(X_test)
                
                # Métricas de clasificación
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred
                }
                
                print(f"\n📊 {model_name}:")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                
            else:
                # Para modelos no supervisados (detección de anomalías)
                y_pred = model.fit_predict(X_test)
                # Convertir -1 a 1 (anomalía) y 1 a 0 (normal) para algunos modelos
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
                
                print(f"\n📊 {model_name} (No supervisado):")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                
        except Exception as e:
            print(f"❌ Error evaluando {model_name}: {e}")
            self.results[model_name] = {'error': str(e)}
    
    def build_autoencoder(self, input_dim):
        """
        Construye un autoencoder para detección de anomalías
        """
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(8, activation='relu')(input_layer)
        encoder = Dense(4, activation='relu')(encoder)
        encoder = Dense(2, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(4, activation='relu')(encoder)
        decoder = Dense(8, activation='relu')(decoder)
        decoder = Dense(input_dim, activation='linear')(decoder)
        
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def run_all_models(self):
        """
        Ejecuta todos los modelos de ML requeridos
        """
        print("\n🚀 Iniciando análisis con 10 modelos de Machine Learning...")
        print("📋 Modelos: Isolation Forest, One-Class SVM, Random Forest, k-NN, DBSCAN, LOF, MLP, AdaBoost, Gradient Boosting, SVM")
        
        # 1. Isolation Forest (Detección de anomalías)
        print("\n1️⃣ Isolation Forest")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(self.X_train)
        y_pred_iso = iso_forest.predict(self.X_test)
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convertir -1 a 1 (anomalía)
        
        accuracy = accuracy_score(self.y_test, y_pred_iso)
        f1 = f1_score(self.y_test, y_pred_iso, average='weighted')
        self.results['Isolation Forest'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_iso
        }
        print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 2. One-Class SVM (Detección de anomalías)
        print("\n2️⃣ One-Class SVM")
        oc_svm = OneClassSVM(gamma='scale', nu=0.1)
        oc_svm.fit(self.X_train)
        y_pred_svm = oc_svm.predict(self.X_test)
        y_pred_svm = np.where(y_pred_svm == -1, 1, 0)
        
        accuracy = accuracy_score(self.y_test, y_pred_svm)
        f1 = f1_score(self.y_test, y_pred_svm, average='weighted')
        self.results['One-Class SVM'] = {
            'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_svm
        }
        print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # 3. Autoencoder (si TensorFlow está disponible)
        if TENSORFLOW_AVAILABLE:
            print("\n3️⃣ Autoencoder")
            autoencoder = self.build_autoencoder(self.X_train.shape[1])
            autoencoder.fit(self.X_train, self.X_train, epochs=50, batch_size=16, verbose=0)
            
            # Calcular error de reconstrucción
            X_test_pred = autoencoder.predict(self.X_test)
            mse = np.mean(np.power(self.X_test - X_test_pred, 2), axis=1)
            threshold = np.percentile(mse, 90)  # Top 10% como anomalías
            y_pred_ae = (mse > threshold).astype(int)
            
            accuracy = accuracy_score(self.y_test, y_pred_ae)
            f1 = f1_score(self.y_test, y_pred_ae, average='weighted')
            self.results['Autoencoder'] = {
                'accuracy': accuracy, 'f1_score': f1, 'predictions': y_pred_ae
            }
            print(f"   Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        else:
            print("\n3️⃣ Autoencoder - ⚠️ Saltado (TensorFlow no disponible)")
        
        # 4. Random Forest (Clasificación)
        print("\n4️⃣ Random Forest")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.evaluate_model(rf, self.X_test, self.y_test, 'Random Forest')
        
        # 5. k-NN (Clasificación)
        print("\n5️⃣ k-Nearest Neighbors")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        self.evaluate_model(knn, self.X_test, self.y_test, 'k-NN')
        
        # 6. DBSCAN (Clustering para anomalías)
        print("\n6️⃣ DBSCAN")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(self.X_test)
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
