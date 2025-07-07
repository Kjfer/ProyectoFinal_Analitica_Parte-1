#!/usr/bin/env python3
# setup_ml_environment.py
"""
Script para configurar el entorno de Machine Learning para DryWall Alert
"""

import subprocess
import sys
import os

def install_requirements():
    """
    Instala todas las dependencias necesarias
    """
    print("📦 Instalando dependencias de Machine Learning...")
    
    try:
        # Actualizar pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Instalar requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencias instaladas exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def test_imports():
    """
    Prueba que todas las librerías se importen correctamente
    """
    print("🧪 Probando importaciones...")
    
    required_modules = [
        "pandas", "numpy", "matplotlib", "seaborn",
        "sklearn", "tensorflow"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Módulos faltantes: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ Todas las librerías están disponibles")
        return True

def setup_project_structure():
    """
    Crea la estructura de directorios necesaria
    """
    print("📁 Configurando estructura del proyecto...")
    
    directories = [
        "ml_results",
        "ml_models", 
        "ml_reports",
        "data_exports"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Creado: {directory}/")
        else:
            print(f"📁 Ya existe: {directory}/")

def create_jupyter_notebook():
    """
    Crea un notebook de Jupyter para análisis interactivo
    """
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# DryWall Alert - Análisis de Machine Learning\n",
                    "\n",
                    "Este notebook contiene el análisis completo de modelos de ML para detección de anomalías en sensores de humedad.\n",
                    "\n",
                    "## Grupo 3 - DryWall Alert\n",
                    "**Problema**: Detección de anomalías / Clasificación  \n",
                    "**Modelos**: Isolation Forest, One-Class SVM, Autoencoder, Random Forest, kNN, DBSCAN, LOF, MLP, AdaBoost, Gradient Boost"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Importar librerías necesarias\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from ml_analysis import DryWallAnalyzer\n",
                    "\n",
                    "# Configurar visualización\n",
                    "plt.style.use('seaborn-v0_8')\n",
                    "sns.set_palette('husl')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Inicializar el analizador\n",
                    "analyzer = DryWallAnalyzer('humedad_datos.csv')\n",
                    "print(\"🚀 Analizador inicializado\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cargar y explorar datos\n",
                    "analyzer.load_and_prepare_data()\n",
                    "analyzer.split_data()\n",
                    "\n",
                    "# Mostrar información básica\n",
                    "print(\"📊 Datos cargados:\")\n",
                    "print(f\"   Total registros: {len(analyzer.df)}\")\n",
                    "print(f\"   Features: {list(analyzer.X.columns)}\")\n",
                    "print(f\"   Distribución de clases:\")\n",
                    "print(analyzer.df['is_anomaly'].value_counts())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualización de datos\n",
                    "analyzer.visualize_data()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Ejecutar todos los modelos de ML\n",
                    "analyzer.run_all_models()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Comparar rendimiento de modelos\n",
                    "comparison_df = analyzer.compare_models()\n",
                    "print(\"\\n📊 Mejores modelos:\")\n",
                    "print(comparison_df.head())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Generar reporte final\n",
                    "analyzer.generate_report()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Conclusiones\n",
                    "\n",
                    "### Análisis de Resultados:\n",
                    "\n",
                    "1. **Mejor Modelo**: [Se completará automáticamente]\n",
                    "2. **Métricas de Rendimiento**: [Se completará automáticamente]\n",
                    "3. **Justificación**: [Se completará automáticamente]\n",
                    "\n",
                    "### Implementación en Producción:\n",
                    "\n",
                    "El modelo seleccionado se puede integrar con el sistema de alertas de WhatsApp existente para mejorar la precisión de detección de filtraciones.\n",
                    "\n",
                    "### Próximos Pasos:\n",
                    "\n",
                    "1. Recopilar más datos para mejorar el entrenamiento\n",
                    "2. Implementar validación cruzada temporal\n",
                    "3. Ajustar hiperparámetros del mejor modelo\n",
                    "4. Integrar sistema de retroalimentación"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    
    with open('ml_analysis_notebook.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    
    print("📓 Notebook creado: ml_analysis_notebook.ipynb")

def main():
    """
    Función principal de configuración
    """
    print("🏠 DRYWALL ALERT - Configuración ML")
    print("=" * 50)
    
    # Paso 1: Instalar dependencias
    if not install_requirements():
        print("❌ Error en instalación. Revisar dependencias.")
        return False
    
    # Paso 2: Probar importaciones
    if not test_imports():
        print("❌ Error en importaciones. Revisar instalación.")
        return False
    
    # Paso 3: Configurar estructura
    setup_project_structure()
    
    # Paso 4: Crear notebook
    create_jupyter_notebook()
    
    print("\n✅ CONFIGURACIÓN COMPLETADA")
    print("\n📋 Próximos pasos:")
    print("1. Ejecutar: python ml_analysis.py")
    print("2. O abrir: ml_analysis_notebook.ipynb")
    print("3. Para integración: python integrated_ml_system.py integrate")
    
    return True

if __name__ == "__main__":
    main()
