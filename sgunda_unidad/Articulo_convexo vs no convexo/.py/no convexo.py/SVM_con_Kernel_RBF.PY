# Implementación de SVM con Kernel RBF para Dataset de Cáncer de Mama Wisconsin
# Proyecto: Técnicas de Optimización Convexa y No Convexa

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
print("IMPLEMENTACIÓN DE SVM CON KERNEL RBF")
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

print(f"\n🔮 CONFIGURACIÓN DE SVM RBF:")
print("• Kernel: Radial Basis Function (RBF)")
print("• Función del kernel: K(x,z) = exp(-γ||x-z||²)")
print("• Mapeo: Espacio de características de dimensión infinita")
print("• Frontera de decisión: No lineal en espacio original")
print("• Optimización: No convexa debido al kernel no lineal")

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

# Estandarización (CRÍTICA para SVM RBF)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"• Estandarización aplicada ✓ (CRÍTICA para kernel RBF)")
print(f"• Media de características después de escalar: {np.mean(X_train_scaled, axis=0)[:3].round(3)}")
print(f"• Desviación estándar después de escalar: {np.std(X_train_scaled, axis=0)[:3].round(3)}")

# =============================================================================
# 3. OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================

print(f"\n⚙️ OPTIMIZACIÓN DE HIPERPARÁMETROS:")

# Definir grid de búsqueda para C y gamma (parámetros críticos del RBF)
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0, 1000.0],        # Parámetro de regularización
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10.0]  # Parámetro del kernel RBF
}

print(f"• Valores de C a evaluar: {param_grid['C']}")
print(f"• Valores de gamma a evaluar: {param_grid['gamma']}")
print(f"• Configuraciones totales: {len(param_grid['C']) * len(param_grid['gamma'])}")

# Grid Search con validación cruzada
svm_rbf_grid = GridSearchCV(
    SVC(kernel='rbf', random_state=42, probability=True),
    param_grid,
    cv=5,  # 5-fold para mejor evaluación
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print("• Ejecutando Grid Search con validación cruzada 5-fold...")
print("• (Evaluando combinaciones C-gamma para optimización no convexa)")
start_time = time.time()
svm_rbf_grid.fit(X_train_scaled, y_train)
grid_time = time.time() - start_time

print(f"• Mejores hiperparámetros: {svm_rbf_grid.best_params_}")
print(f"• Mejor puntuación F1 (CV): {svm_rbf_grid.best_score_:.4f}")
print(f"• Tiempo de optimización: {grid_time:.2f} segundos")

# Análisis del espacio de hiperparámetros
print(f"\n🎯 ANÁLISIS DEL ESPACIO DE HIPERPARÁMETROS:")
print(f"• C óptimo: {svm_rbf_grid.best_params_['C']}")
print(f"• Gamma óptimo: {svm_rbf_grid.best_params_['gamma']}")

if svm_rbf_grid.best_params_['gamma'] in ['scale', 'auto']:
    if svm_rbf_grid.best_params_['gamma'] == 'scale':
        gamma_value = 1 / (X_train_scaled.shape[1] * X_train_scaled.var())
        print(f"• Gamma 'scale' equivale a: {gamma_value:.6f}")
    else:
        gamma_value = 1 / X_train_scaled.shape[1]
        print(f"• Gamma 'auto' equivale a: {gamma_value:.6f}")
else:
    gamma_value = svm_rbf_grid.best_params_['gamma']
    print(f"• Gamma numérico: {gamma_value}")

# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

print(f"\n🚀 ENTRENAMIENTO DEL MODELO FINAL:")

# Usar los mejores hiperparámetros
best_svm_rbf = svm_rbf_grid.best_estimator_

# Medir tiempo de convergencia
start_time = time.time()
best_svm_rbf.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"• Modelo SVM RBF entrenado con hiperparámetros óptimos")
print(f"• Tiempo de convergencia: {training_time:.4f} segundos")
print(f"• Número de vectores de soporte: {best_svm_rbf.n_support_}")
print(f"• Total de vectores de soporte: {np.sum(best_svm_rbf.n_support_)} de {len(X_train)} muestras ({np.sum(best_svm_rbf.n_support_)/len(X_train)*100:.1f}%)")

# Comparar complejidad con SVM lineal
print(f"• Comparación con SVM Lineal:")
print(f"  - SVM Lineal: 51 vectores de soporte (11.2%)")
print(f"  - SVM RBF: {np.sum(best_svm_rbf.n_support_)} vectores de soporte ({np.sum(best_svm_rbf.n_support_)/len(X_train)*100:.1f}%)")

# =============================================================================
# 5. EVALUACIÓN DEL MODELO
# =============================================================================

print(f"\n📊 EVALUACIÓN DEL MODELO:")

# Predicciones
y_pred = best_svm_rbf.predict(X_test_scaled)
y_pred_proba = best_svm_rbf.predict_proba(X_test_scaled)[:, 1]

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

# Análisis de vectores de soporte
print(f"\n🎯 ANÁLISIS DE VECTORES DE SOPORTE:")
print(f"• Vectores de soporte por clase: {best_svm_rbf.n_support_}")
print(f"• Porcentaje total: {np.sum(best_svm_rbf.n_support_)/len(X_train)*100:.1f}%")

if np.sum(best_svm_rbf.n_support_)/len(X_train) > 0.3:
    print("• Interpretación: Frontera de decisión compleja (muchos vectores necesarios)")
elif np.sum(best_svm_rbf.n_support_)/len(X_train) > 0.15:
    print("• Interpretación: Frontera de decisión moderadamente compleja")
else:
    print("• Interpretación: Frontera de decisión relativamente simple")

# Análisis de la función del kernel
print(f"\n🔮 ANÁLISIS DEL KERNEL RBF:")
print(f"• Parámetro gamma: {svm_rbf_grid.best_params_['gamma']}")
if isinstance(svm_rbf_grid.best_params_['gamma'], str):
    gamma_val = gamma_value
else:
    gamma_val = svm_rbf_grid.best_params_['gamma']

if gamma_val > 1:
    print("• Comportamiento: Kernel muy localizado (alta varianza, bajo sesgo)")
elif gamma_val > 0.1:
    print("• Comportamiento: Kernel moderadamente localizado")
else:
    print("• Comportamiento: Kernel suave (baja varianza, alto sesgo)")

# =============================================================================
# 7. VALIDACIÓN CRUZADA ADICIONAL
# =============================================================================

print(f"\n✅ VALIDACIÓN CRUZADA FINAL:")

# Validación cruzada con múltiples métricas
cv_accuracy = cross_val_score(best_svm_rbf, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_precision = cross_val_score(best_svm_rbf, X_train_scaled, y_train, cv=5, scoring='precision')
cv_recall = cross_val_score(best_svm_rbf, X_train_scaled, y_train, cv=5, scoring='recall')
cv_f1 = cross_val_score(best_svm_rbf, X_train_scaled, y_train, cv=5, scoring='f1')

print(f"• Accuracy CV (5-fold): {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"• Precision CV (5-fold): {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
print(f"• Recall CV (5-fold): {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"• F1-Score CV (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# =============================================================================
# 8. COMPARACIÓN CON MÉTODOS ANTERIORES
# =============================================================================

print(f"\n🆚 COMPARACIÓN CON MÉTODOS ANTERIORES:")
print("(Valores de referencia)")
print(f"• SVM Lineal - Accuracy: 0.982")
print(f"• Regresión Logística - Accuracy: 0.974")
print(f"• Redes Neuronales - Accuracy: 0.956")
print(f"• Regresión Ridge - Accuracy: 0.956")
print(f"• SVM RBF - Accuracy: {accuracy:.3f}")

# Comparación específica con SVM Lineal
print(f"\n🔄 SVM LINEAL vs SVM RBF:")
print(f"• Accuracy: 0.982 vs {accuracy:.3f} ({'🏆 RBF mejor' if accuracy > 0.982 else '🏆 Lineal mejor' if accuracy < 0.982 else '🤝 Empate'})")
print(f"• Vectores de soporte: 51 (11.2%) vs {np.sum(best_svm_rbf.n_support_)} ({np.sum(best_svm_rbf.n_support_)/len(X_train)*100:.1f}%)")
print(f"• Tiempo: 0.0065s vs {training_time:.4f}s")

# =============================================================================
# 9. RESUMEN FINAL PARA EL PAPER
# =============================================================================

print(f"\n" + "="*60)
print("RESUMEN PARA EL PAPER - SVM RBF")
print("="*60)

print(f"\n📊 RESULTADOS FINALES:")
print(f"• Hiperparámetros óptimos:")
print(f"  - C: {svm_rbf_grid.best_params_['C']}")
print(f"  - Gamma: {svm_rbf_grid.best_params_['gamma']}")
print(f"• Precisión (Accuracy): {accuracy:.3f}")
print(f"• Precisión (Precision): {precision:.3f}")
print(f"• Sensibilidad (Recall): {recall:.3f}")
print(f"• Puntuación F1: {f1:.3f}")
print(f"• AUC-ROC: {auc_roc:.3f}")
print(f"• Tiempo de Convergencia: {training_time:.4f}s")
print(f"• Vectores de Soporte: {np.sum(best_svm_rbf.n_support_)} ({np.sum(best_svm_rbf.n_support_)/len(X_train)*100:.1f}%)")

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

print(f"\n✨ VENTAJAS DE SVM RBF:")
print("• Capacidad de modelar fronteras de decisión no lineales")
print("• Mapeo implícito a espacio de alta dimensión")
print("• Robusto ante valores atípicos")
print("• Flexible con parámetro gamma")

print(f"\n⚠️ CONSIDERACIONES NO CONVEXAS:")
print("• Optimización no convexa debido al kernel no lineal")
print("• Sensible a la selección de hiperparámetros C y gamma")
print("• Mayor complejidad computacional que SVM lineal")
print("• Riesgo de sobreajuste con gamma alto")

print(f"\n🔧 COMPLEJIDAD DEL MODELO:")
complexity_ratio = np.sum(best_svm_rbf.n_support_) / 51  # Comparado con SVM lineal
print(f"• Complejidad relativa: {complexity_ratio:.1f}x vs SVM Lineal")
print(f"• Tiempo relativo: {training_time/0.0065:.1f}x vs SVM Lineal")

print(f"\n🎭 COMPARACIÓN KERNEL LINEAL vs RBF:")
if accuracy > 0.982:
    print("• El kernel RBF mejoró el rendimiento - dataset beneficia de no linealidad")
elif accuracy == 0.982:
    print("• Rendimiento equivalente - dataset es linealmente separable")
else:
    print("• El kernel lineal fue mejor - dataset no requiere modelado no lineal")

print(f"\n" + "="*60)
print("¡Implementación de SVM RBF completada!")
print("="*60)