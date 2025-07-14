# Implementación de Regresión Ridge para Dataset de Cáncer de Mama Wisconsin
# Proyecto: Técnicas de Optimización Convexa y No Convexa

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# =============================================================================

print("="*60)
print("IMPLEMENTACIÓN DE REGRESIÓN RIDGE")
print("Dataset: Wisconsin Breast Cancer")
print("="*60)

# Cargar el dataset
data = load_breast_cancer()
X = data.data  # Características (30 features)
y = data.target  # Etiquetas (0=maligno, 1=benigno)

print(f"\n📊 INFORMACIÓN DEL DATASET:")
print(f"• Número de muestras: {X.shape[0]}")
print(f"• Número de características: {X.shape[1]}")
print(f"• Clases originales: {data.target_names}")
print(f"• Para Ridge: Convertimos a problema de regresión")
print(f"• Valores objetivo: 0 (maligno) → 1 (benigno)")

# =============================================================================
# 2. PREPROCESAMIENTO DE DATOS
# =============================================================================

print(f"\n🔧 PREPROCESAMIENTO:")

# División entrenamiento/prueba (80/20) - MISMA DIVISIÓN QUE ANTERIORES
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
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
}

# Para Ridge usaremos R² como métrica de optimización, luego convertiremos a clasificación
ridge_grid = GridSearchCV(
    Ridge(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',  # R² para regresión
    n_jobs=-1,
    verbose=0
)

print("• Ejecutando Grid Search con validación cruzada 5-fold...")
start_time = time.time()
ridge_grid.fit(X_train_scaled, y_train)
grid_time = time.time() - start_time

print(f"• Mejores hiperparámetros: {ridge_grid.best_params_}")
print(f"• Mejor puntuación R² (CV): {ridge_grid.best_score_:.4f}")
print(f"• Tiempo de optimización: {grid_time:.2f} segundos")

# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

print(f"\n🚀 ENTRENAMIENTO DEL MODELO FINAL:")

# Usar los mejores hiperparámetros
best_ridge = ridge_grid.best_estimator_

# Medir tiempo de convergencia
start_time = time.time()
best_ridge.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"• Modelo Ridge entrenado con hiperparámetros óptimos")
print(f"• Tiempo de convergencia: {training_time:.4f} segundos")
print(f"• Coeficiente de determinación R²: {best_ridge.score(X_train_scaled, y_train):.4f}")

# =============================================================================
# 5. PREDICCIONES Y CONVERSIÓN A CLASIFICACIÓN
# =============================================================================

print(f"\n🔄 CONVERSIÓN DE REGRESIÓN A CLASIFICACIÓN:")

# Predicciones continuas
y_pred_continuous = best_ridge.predict(X_test_scaled)
print(f"• Predicciones continuas - Rango: [{y_pred_continuous.min():.3f}, {y_pred_continuous.max():.3f}]")
print(f"• Media de predicciones: {y_pred_continuous.mean():.3f}")

# Convertir a clasificación binaria usando umbral 0.5
threshold = 0.5
y_pred = (y_pred_continuous >= threshold).astype(int)

print(f"• Umbral de clasificación: {threshold}")
print(f"• Conversión: valores ≥ {threshold} → clase 1 (benigno)")
print(f"• Conversión: valores < {threshold} → clase 0 (maligno)")

# =============================================================================
# 6. EVALUACIÓN DEL MODELO
# =============================================================================

print(f"\n📊 EVALUACIÓN DEL MODELO:")

# Calcular métricas de regresión
mse = mean_squared_error(y_test, y_pred_continuous)
r2 = r2_score(y_test, y_pred_continuous)

print(f"\n📈 MÉTRICAS DE REGRESIÓN:")
print(f"• Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"• Coeficiente de Determinación (R²): {r2:.4f}")

# Calcular métricas de clasificación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Para AUC-ROC usamos las predicciones continuas como probabilidades
# Normalizar las predicciones continuas a [0,1] para simular probabilidades
y_pred_proba_normalized = (y_pred_continuous - y_pred_continuous.min()) / (y_pred_continuous.max() - y_pred_continuous.min())
auc_roc = roc_auc_score(y_test, y_pred_proba_normalized)

print(f"\n📈 MÉTRICAS DE CLASIFICACIÓN:")
print(f"• Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"• Precisión (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"• Sensibilidad (Recall): {recall:.4f} ({recall*100:.2f}%)")
print(f"• Puntuación F1: {f1:.4f} ({f1*100:.2f}%)")
print(f"• AUC-ROC: {auc_roc:.4f} ({auc_roc*100:.2f}%)")
print(f"• Tiempo de Convergencia: {training_time:.4f} segundos")

# =============================================================================
# 7. ANÁLISIS DETALLADO
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

# Análisis de la distribución de predicciones
print(f"\n📊 DISTRIBUCIÓN DE PREDICCIONES CONTINUAS:")
print(f"• Predicciones para clase 0 (maligno):")
malignant_predictions = y_pred_continuous[y_test == 0]
print(f"  - Media: {malignant_predictions.mean():.3f}")
print(f"  - Desv. Estándar: {malignant_predictions.std():.3f}")
print(f"  - Rango: [{malignant_predictions.min():.3f}, {malignant_predictions.max():.3f}]")

print(f"• Predicciones para clase 1 (benigno):")
benign_predictions = y_pred_continuous[y_test == 1]
print(f"  - Media: {benign_predictions.mean():.3f}")
print(f"  - Desv. Estándar: {benign_predictions.std():.3f}")
print(f"  - Rango: [{benign_predictions.min():.3f}, {benign_predictions.max():.3f}]")

# Características más importantes
feature_importance = np.abs(best_ridge.coef_)
feature_names = data.feature_names
top_features_idx = np.argsort(feature_importance)[-10:][::-1]

print(f"\n🔝 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
for i, idx in enumerate(top_features_idx):
    print(f"  {i+1:2d}. {feature_names[idx]:<25} | Coeficiente: {best_ridge.coef_[idx]:7.3f}")

# =============================================================================
# 8. VALIDACIÓN CRUZADA ADICIONAL
# =============================================================================

print(f"\n✅ VALIDACIÓN CRUZADA FINAL:")

# Para validación cruzada, necesitamos crear una función que convierta regresión a clasificación
def ridge_classification_score(estimator, X, y):
    y_pred_cont = estimator.predict(X)
    y_pred_class = (y_pred_cont >= 0.5).astype(int)
    return f1_score(y, y_pred_class)

# Validación cruzada con conversión a clasificación
cv_scores = cross_val_score(best_ridge, X_train_scaled, y_train, cv=5, scoring=ridge_classification_score)
cv_r2 = cross_val_score(best_ridge, X_train_scaled, y_train, cv=5, scoring='r2')

print(f"• F1-Score CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"• R² CV (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# =============================================================================
# 9. COMPARACIÓN CON MÉTODOS ANTERIORES
# =============================================================================

print(f"\n🆚 COMPARACIÓN RÁPIDA:")
print("(Valores de referencia de métodos anteriores)")
print(f"• Regresión Logística - Accuracy: 0.974")
print(f"• SVM Lineal - Accuracy: 0.982")
print(f"• Regresión Ridge - Accuracy: {accuracy:.3f}")

# =============================================================================
# 10. RESUMEN FINAL PARA EL PAPER
# =============================================================================

print(f"\n" + "="*60)
print("RESUMEN PARA EL PAPER - REGRESIÓN RIDGE")
print("="*60)

print(f"\n📊 RESULTADOS FINALES:")
print(f"• Hiperparámetros óptimos: alpha={ridge_grid.best_params_['alpha']}")
print(f"• Precisión (Accuracy): {accuracy:.3f}")
print(f"• Precisión (Precision): {precision:.3f}")
print(f"• Sensibilidad (Recall): {recall:.3f}")
print(f"• Puntuación F1: {f1:.3f}")
print(f"• AUC-ROC: {auc_roc:.3f}")
print(f"• Tiempo de Convergencia: {training_time:.4f}s")
print(f"• R² (como regresión): {r2:.3f}")

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

print(f"\n✨ VENTAJAS DE REGRESIÓN RIDGE:")
print("• Previene sobreajuste con regularización L2")
print("• Maneja bien la multicolinealidad")
print("• Solución analítica directa (muy rápida)")
print("• Estable numéricamente")

print(f"\n📊 CONSIDERACIONES ESPECIALES:")
print("• Adaptada de regresión a clasificación")
print("• Umbral de 0.5 para conversión binaria")
print("• Predicciones continuas proporcionan información adicional")

print(f"\n🔄 SEPARACIÓN DE CLASES:")
print(f"• Predicciones promedio para malignos: {malignant_predictions.mean():.3f}")
print(f"• Predicciones promedio para benignos: {benign_predictions.mean():.3f}")
print(f"• Separación: {abs(benign_predictions.mean() - malignant_predictions.mean()):.3f}")

print(f"\n" + "="*60)
print("¡Implementación de Regresión Ridge completada!")
print("="*60)