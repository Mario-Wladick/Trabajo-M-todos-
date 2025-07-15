# Implementación de SVM Lineal para Dataset de Cáncer de Mama Wisconsin
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

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# =============================================================================

print("="*60)
print("IMPLEMENTACIÓN DE SVM LINEAL")
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

# =============================================================================
# 2. PREPROCESAMIENTO DE DATOS
# =============================================================================

print(f"\n🔧 PREPROCESAMIENTO:")

# División entrenamiento/prueba (80/20) - MISMA DIVISIÓN QUE REGRESIÓN LOGÍSTICA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"• Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"• Conjunto de prueba: {X_test.shape[0]} muestras")

# Estandarización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"• Estandarización aplicada ✓")
print(f"• Media de características después de escalar: {np.mean(X_train_scaled, axis=0)[:3].round(3)}")
print(f"• Desviación estándar después de escalar: {np.std(X_train_scaled, axis=0)[:3].round(3)}")

# =============================================================================
# 3. OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================

print(f"\n⚙️ OPTIMIZACIÓN DE HIPERPARÁMETROS:")

# Definir grid de búsqueda para el parámetro de regularización
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'gamma': ['scale', 'auto']  # Para kernel lineal no es tan relevante, pero lo incluimos
}

# Grid Search con validación cruzada
svm_grid = GridSearchCV(
    SVC(kernel='linear', random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print("• Ejecutando Grid Search con validación cruzada 5-fold...")
start_time = time.time()
svm_grid.fit(X_train_scaled, y_train)
grid_time = time.time() - start_time

print(f"• Mejores hiperparámetros: {svm_grid.best_params_}")
print(f"• Mejor puntuación F1 (CV): {svm_grid.best_score_:.4f}")
print(f"• Tiempo de optimización: {grid_time:.2f} segundos")

# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

print(f"\n🚀 ENTRENAMIENTO DEL MODELO FINAL:")

# Usar los mejores hiperparámetros
best_svm = svm_grid.best_estimator_

# Medir tiempo de convergencia
start_time = time.time()
best_svm.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"• Modelo SVM Lineal entrenado con hiperparámetros óptimos")
print(f"• Tiempo de convergencia: {training_time:.4f} segundos")
print(f"• Número de vectores de soporte: {best_svm.n_support_}")
print(f"• Total de vectores de soporte: {np.sum(best_svm.n_support_)} de {len(X_train)} muestras ({np.sum(best_svm.n_support_)/len(X_train)*100:.1f}%)")

# =============================================================================
# 5. EVALUACIÓN DEL MODELO
# =============================================================================

print(f"\n📊 EVALUACIÓN DEL MODELO:")

# Predicciones
y_pred = best_svm.predict(X_test_scaled)
y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]

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
print(f"• Vectores de soporte por clase: {best_svm.n_support_}")
print(f"• Porcentaje de muestras que son vectores de soporte: {np.sum(best_svm.n_support_)/len(X_train)*100:.1f}%")
print(f"• Esto indica qué tan compleja es la frontera de decisión")

# Coeficientes del hiperplano (para kernel lineal)
if hasattr(best_svm, 'coef_'):
    feature_importance = np.abs(best_svm.coef_[0])
    feature_names = data.feature_names
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    print(f"\n🔝 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    for i, idx in enumerate(top_features_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:<25} | Coeficiente: {best_svm.coef_[0][idx]:7.3f}")

# =============================================================================
# 7. VALIDACIÓN CRUZADA ADICIONAL
# =============================================================================

print(f"\n✅ VALIDACIÓN CRUZADA FINAL:")

# Validación cruzada con múltiples métricas
cv_accuracy = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_precision = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring='precision')
cv_recall = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring='recall')
cv_f1 = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring='f1')

print(f"• Accuracy CV (5-fold): {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"• Precision CV (5-fold): {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
print(f"• Recall CV (5-fold): {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"• F1-Score CV (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# =============================================================================
# 8. COMPARACIÓN CON REGRESIÓN LOGÍSTICA
# =============================================================================

print(f"\n🆚 COMPARACIÓN RÁPIDA:")
print("(Valores de referencia de Regresión Logística)")
print(f"• Regresión Logística - Accuracy: 0.974")
print(f"• SVM Lineal - Accuracy: {accuracy:.3f}")
print(f"• Diferencia: {accuracy - 0.974:+.3f}")

# =============================================================================
# 9. RESUMEN FINAL PARA EL PAPER
# =============================================================================

print(f"\n" + "="*60)
print("RESUMEN PARA EL PAPER - SVM LINEAL")
print("="*60)

print(f"\n📊 RESULTADOS FINALES:")
print(f"• Hiperparámetros óptimos: C={svm_grid.best_params_['C']}, gamma='{svm_grid.best_params_['gamma']}'")
print(f"• Precisión (Accuracy): {accuracy:.3f}")
print(f"• Precisión (Precision): {precision:.3f}")
print(f"• Sensibilidad (Recall): {recall:.3f}")
print(f"• Puntuación F1: {f1:.3f}")
print(f"• AUC-ROC: {auc_roc:.3f}")
print(f"• Tiempo de Convergencia: {training_time:.4f}s")
print(f"• Vectores de Soporte: {np.sum(best_svm.n_support_)} ({np.sum(best_svm.n_support_)/len(X_train)*100:.1f}%)")

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

print(f"\n✨ VENTAJAS DE SVM LINEAL:")
print("• Frontera de decisión óptima (maximiza margen)")
print("• Robusto ante valores atípicos")
print("• Eficiente en memoria (solo almacena vectores de soporte)")
print("• Base teórica sólida")

print(f"\n⚡ EFICIENCIA:")
if np.sum(best_svm.n_support_)/len(X_train) < 0.3:
    print("• Modelo eficiente: pocos vectores de soporte necesarios")
else:
    print("• Modelo complejo: muchos vectores de soporte necesarios")

print(f"\n" + "="*60)
print("¡Implementación de SVM Lineal completada!")
print("="*60)