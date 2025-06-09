# Implementación de Redes Neuronales para Dataset de Cáncer de Mama Wisconsin
# Proyecto: Técnicas de Optimización Convexa y No Convexa

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# =============================================================================

print("="*60)
print("IMPLEMENTACIÓN DE REDES NEURONALES")
print("Dataset: Wisconsin Breast Cancer")
print("="*60)

# Cargar el dataset
data = load_breast_cancer()
X = data.data  # Características (30 features)
y = data.target  # Etiquetas (0=maligno, 1=benigno)

print(f"\n📊 INFORMACIÓN DEL DATASET:")
print(f"• Número de muestras: {X.shape[0]}")
print(f"• Número de características: {X.shape[1]}")
print(f"• Clases: {data.target_names}")
print(f"• Distribución de clases:")
unique, counts = np.unique(y, return_counts=True)
for i, (clase, count) in enumerate(zip(data.target_names, counts)):
    print(f"  - {clase}: {count} ({count/len(y)*100:.1f}%)")

print(f"\n🧠 CONFIGURACIÓN DE RED NEURONAL:")
print("• Arquitectura: MultiLayer Perceptron (MLP)")
print("• Capas de entrada: 30 neuronas (features)")
print("• Capas ocultas: Variable (hiperparámetro)")
print("• Capa de salida: 2 neuronas (clasificación binaria)")
print("• Función de activación: ReLU (capas ocultas), Softmax (salida)")

# =============================================================================
# 2. PREPROCESAMIENTO DE DATOS
# =============================================================================

print(f"\n🔧 PREPROCESAMIENTO:")

# División entrenamiento/prueba (80/20) - MISMA DIVISIÓN QUE MÉTODOS ANTERIORES
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"• Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"• Conjunto de prueba: {X_test.shape[0]} muestras")

# Estandarización (CRÍTICA para redes neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"• Estandarización aplicada ✓ (CRÍTICA para redes neuronales)")
print(f"• Media de características después de escalar: {np.mean(X_train_scaled, axis=0)[:3].round(3)}")
print(f"• Desviación estándar después de escalar: {np.std(X_train_scaled, axis=0)[:3].round(3)}")

# =============================================================================
# 3. OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================

print(f"\n⚙️ OPTIMIZACIÓN DE HIPERPARÁMETROS:")

# Definir grid de búsqueda para arquitectura y regularización
param_grid = {
    'hidden_layer_sizes': [
        (50,),          # 1 capa con 50 neuronas
        (100,),         # 1 capa con 100 neuronas
        (50, 25),       # 2 capas: 50 y 25 neuronas
        (100, 50),      # 2 capas: 100 y 50 neuronas
        (100, 50, 25)   # 3 capas: 100, 50 y 25 neuronas
    ],
    'alpha': [0.0001, 0.001, 0.01],  # Regularización L2
    'learning_rate_init': [0.001, 0.01]  # Tasa de aprendizaje inicial
}

print(f"• Arquitecturas a evaluar: {len(param_grid['hidden_layer_sizes'])}")
print(f"• Configuraciones totales: {len(param_grid['hidden_layer_sizes']) * len(param_grid['alpha']) * len(param_grid['learning_rate_init'])}")

# Grid Search con validación cruzada
mlp_grid = GridSearchCV(
    MLPClassifier(
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ),
    param_grid,
    cv=3,  # Reducido para redes neuronales (más costoso)
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print("• Ejecutando Grid Search con validación cruzada 3-fold...")
print("• (Reducido a 3-fold debido al costo computacional de redes neuronales)")
start_time = time.time()
mlp_grid.fit(X_train_scaled, y_train)
grid_time = time.time() - start_time

print(f"• Mejores hiperparámetros: {mlp_grid.best_params_}")
print(f"• Mejor puntuación F1 (CV): {mlp_grid.best_score_:.4f}")
print(f"• Tiempo de optimización: {grid_time:.2f} segundos")

# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

print(f"\n🚀 ENTRENAMIENTO DEL MODELO FINAL:")

# Usar los mejores hiperparámetros
best_mlp = mlp_grid.best_estimator_

# Reentrenar con datos completos de entrenamiento
start_time = time.time()
best_mlp.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"• Red neuronal entrenada con hiperparámetros óptimos")
print(f"• Tiempo de convergencia: {training_time:.4f} segundos")
print(f"• Arquitectura final: {best_mlp.hidden_layer_sizes}")
print(f"• Número de iteraciones: {best_mlp.n_iter_}")
print(f"• Función de pérdida final: {best_mlp.loss_:.6f}")

# Información de la arquitectura
total_params = 0
layer_sizes = [X_train_scaled.shape[1]] + list(best_mlp.hidden_layer_sizes) + [len(np.unique(y_train))]
for i in range(len(layer_sizes)-1):
    params_layer = layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]  # pesos + sesgos
    total_params += params_layer
    print(f"• Capa {i+1}: {layer_sizes[i]} → {layer_sizes[i+1]} ({params_layer} parámetros)")

print(f"• Total de parámetros en la red: {total_params}")

# =============================================================================
# 5. EVALUACIÓN DEL MODELO
# =============================================================================

print(f"\n📊 EVALUACIÓN DEL MODELO:")

# Predicciones
y_pred = best_mlp.predict(X_test_scaled)
y_pred_proba = best_mlp.predict_proba(X_test_scaled)[:, 1]

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Mostrar resultados
print(f"\n📈 MÉTRICAS DE RENDIMIENTO:")
print(f"• Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"• Precisión (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"• Sensibilidad (Recall): {recall:.4f} ({recall*100:.2f}%)")
print(f"• Puntuación F1: {f1:.4f} ({f1*100:.2f}%)")
print(f"• AUC-ROC: {auc_roc:.4f} ({auc_roc*100:.2f}%)")
print(f"• Tiempo de Convergencia: {training_time:.4f} segundos")

# =============================================================================
# 6. ANÁLISIS DETALLADO
# =============================================================================

print(f"\n🔍 ANÁLISIS DETALLADO:")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\n📋 MATRIZ DE CONFUSIÓN:")
print(f"                Predicho")
print(f"              Maligno  Benigno")
print(f"Real Maligno     {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"     Benigno     {cm[1,0]:3d}     {cm[1,1]:3d}")

# Calcular falsos positivos y negativos
tn, fp, fn, tp = cm.ravel()
print(f"\n📊 DESGLOSE DE PREDICCIONES:")
print(f"• Verdaderos Positivos (TP): {tp}")
print(f"• Verdaderos Negativos (TN): {tn}")
print(f"• Falsos Positivos (FP): {fp}")
print(f"• Falsos Negativos (FN): {fn}")

# Análisis de la confianza de predicciones
print(f"\n🎯 ANÁLISIS DE CONFIANZA:")
confidence_malignant = y_pred_proba[y_test == 0]  # Probabilidades para casos malignos reales
confidence_benign = y_pred_proba[y_test == 1]    # Probabilidades para casos benignos reales

print(f"• Confianza promedio en casos malignos: {1-confidence_malignant.mean():.3f}")
print(f"• Confianza promedio en casos benignos: {confidence_benign.mean():.3f}")
print(f"• Desviación estándar confianza malignos: {confidence_malignant.std():.3f}")
print(f"• Desviación estándar confianza benignos: {confidence_benign.std():.3f}")

# Información sobre convergencia
print(f"\n🔄 ANÁLISIS DE CONVERGENCIA:")
print(f"• ¿Convergió?: {'Sí' if best_mlp.n_iter_ < best_mlp.max_iter else 'No (máx iteraciones)'}")
print(f"• Iteraciones usadas: {best_mlp.n_iter_} de {best_mlp.max_iter}")
print(f"• Pérdida final: {best_mlp.loss_:.6f}")

# =============================================================================
# 7. VALIDACIÓN CRUZADA ADICIONAL
# =============================================================================

print(f"\n✅ VALIDACIÓN CRUZADA FINAL:")

# Validación cruzada con múltiples métricas (3-fold para redes neuronales)
cv_accuracy = cross_val_score(best_mlp, X_train_scaled, y_train, cv=3, scoring='accuracy')
cv_precision = cross_val_score(best_mlp, X_train_scaled, y_train, cv=3, scoring='precision')
cv_recall = cross_val_score(best_mlp, X_train_scaled, y_train, cv=3, scoring='recall')
cv_f1 = cross_val_score(best_mlp, X_train_scaled, y_train, cv=3, scoring='f1')

print(f"• Accuracy CV (3-fold): {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"• Precision CV (3-fold): {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
print(f"• Recall CV (3-fold): {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"• F1-Score CV (3-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# =============================================================================
# 8. COMPARACIÓN CON MÉTODOS CONVEXOS
# =============================================================================

print(f"\n🆚 COMPARACIÓN CON MÉTODOS CONVEXOS:")
print("(Valores de referencia)")
print(f"• Regresión Logística - Accuracy: 0.974")
print(f"• SVM Lineal - Accuracy: 0.982")
print(f"• Regresión Ridge - Accuracy: 0.956")
print(f"• Redes Neuronales - Accuracy: {accuracy:.3f}")
print(f"• Posición: {'🏆 Mejor' if accuracy > 0.982 else '🥈 Segundo' if accuracy > 0.974 else '🥉 Tercero' if accuracy > 0.956 else '4to lugar'}")

# =============================================================================
# 9. RESUMEN FINAL PARA EL PAPER
# =============================================================================

print(f"\n" + "="*60)
print("RESUMEN PARA EL PAPER - REDES NEURONALES")
print("="*60)

print(f"\n📊 RESULTADOS FINALES:")
print(f"• Arquitectura óptima: {best_mlp.hidden_layer_sizes}")
print(f"• Hiperparámetros óptimos:")
print(f"  - Alpha (regularización): {best_mlp.alpha}")
print(f"  - Learning rate: {best_mlp.learning_rate_init}")
print(f"• Precisión (Accuracy): {accuracy:.3f}")
print(f"• Precisión (Precision): {precision:.3f}")
print(f"• Sensibilidad (Recall): {recall:.3f}")
print(f"• Puntuación F1: {f1:.3f}")
print(f"• AUC-ROC: {auc_roc:.3f}")
print(f"• Tiempo de Convergencia: {training_time:.4f}s")
print(f"• Total de parámetros: {total_params}")
print(f"• Iteraciones de convergencia: {best_mlp.n_iter_}")

print(f"\n🎯 INTERPRETACIÓN CLÍNICA:")
if recall >= 0.95:
    print("• Excelente detección de casos malignos (recall alto)")
elif recall >= 0.90:
    print("• Buena detección de casos malignos")
else:
    print("• Detección moderada de casos malignos")

if precision >= 0.95:
    print("• Muy pocos falsos positivos (precision alta)")
elif precision >= 0.90:
    print("• Pocos falsos positivos")
else:
    print("• Algunos falsos positivos presentes")

print(f"\n✨ VENTAJAS DE REDES NEURONALES:")
print("• Capacidad de modelar relaciones no lineales complejas")
print("• Aproximador universal de funciones")
print("• Flexibilidad en arquitectura")
print("• Probabilidades de salida bien calibradas")

print(f"\n⚠️ CONSIDERACIONES NO CONVEXAS:")
print("• Múltiples óptimos locales posibles")
print("• Sensible a inicialización de pesos")
print("• Requiere más tiempo de entrenamiento")
print("• Mayor riesgo de sobreajuste")

print(f"\n🔧 COMPLEJIDAD COMPUTACIONAL:")
complexity_level = "Baja" if total_params < 1000 else "Media" if total_params < 5000 else "Alta"
print(f"• Complejidad del modelo: {complexity_level} ({total_params} parámetros)")
print(f"• Tiempo relativo: {'Rápido' if training_time < 0.1 else 'Moderado' if training_time < 1.0 else 'Lento'}")

print(f"\n" + "="*60)
print("¡Implementación de Redes Neuronales completada!")
print("="*60)