# TRANSFORMACIÓN LOGARÍTMICA + RANDOM FOREST + OPTUNA
# Solución rápida para datos con outliers extremos

import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. CARGA Y TRANSFORMACIÓN LOGARÍTMICA
# =============================================

def load_and_transform_data():
    """Carga datos y aplica transformación logarítmica"""
    
    print("🔄 TRANSFORMACIÓN LOGARÍTMICA - SOLUCIÓN RÁPIDA")
    print("=" * 60)
    
    # Cargar datos
    csv_path = r'C:\Users\LENOVO\lazypredict_env\DataSet1_Agroindustria_ICA_2023_rev.csv'
    df = pd.read_csv(csv_path, delimiter=';')
    
    print(f"\n📊 DATOS ORIGINALES:")
    print("-" * 30)
    original_target = df['valor_estimado_maximo_venta']
    print(f"Min:    S/ {original_target.min():>15,}")
    print(f"Max:    S/ {original_target.max():>15,}")
    print(f"Media:  S/ {original_target.mean():>15,.0f}")
    print(f"Std:    S/ {original_target.std():>15,.0f}")
    print(f"CV:     {(original_target.std()/original_target.mean())*100:>15.1f}%")
    
    # APLICAR TRANSFORMACIÓN LOGARÍTMICA
    print(f"\n🔄 APLICANDO TRANSFORMACIÓN LOGARÍTMICA...")
    df['log_ventas_maximas'] = np.log(df['valor_estimado_maximo_venta'])
    
    log_target = df['log_ventas_maximas']
    print(f"\n📊 DATOS TRANSFORMADOS (LOG):")
    print("-" * 35)
    print(f"Min:    {log_target.min():>15.2f}")
    print(f"Max:    {log_target.max():>15.2f}")
    print(f"Media:  {log_target.mean():>15.2f}")
    print(f"Std:    {log_target.std():>15.2f}")
    print(f"CV:     {(log_target.std()/log_target.mean())*100:>15.1f}%")
    
    # Verificar mejora en variabilidad
    cv_original = (original_target.std()/original_target.mean())*100
    cv_log = (log_target.std()/log_target.mean())*100
    mejora = cv_original / cv_log
    
    print(f"\n🎯 MEJORA EN VARIABILIDAD:")
    print(f"   CV Original: {cv_original:.1f}%")
    print(f"   CV Log:      {cv_log:.1f}%") 
    print(f"   Mejora:      {mejora:.1f}x mejor")
    
    # Preparar features (igual que antes)
    categorical_features = ['provincia', 'distrito', 'descciiu', 'tamanio_emp']
    encoders = {}
    
    for feature in categorical_features:
        encoders[feature] = LabelEncoder()
        df[f'{feature}_encoded'] = encoders[feature].fit_transform(df[feature])
    
    df['exporta_encoded'] = (df['exporta'] == 'SI').astype(int)
    
    feature_columns = [
        'ciiu', 'provincia_encoded', 'distrito_encoded',
        'descciiu_encoded', 'tamanio_emp_encoded', 'exporta_encoded',
        'valor_estimado_minimo_venta'
    ]
    
    X = df[feature_columns]
    y_log = df['log_ventas_maximas']  # ← USAR LOG TRANSFORM
    
    return X, y_log, df, encoders

# =============================================
# 2. FUNCIÓN OBJETIVO OPTUNA (OPTIMIZADA)
# =============================================

def objective_log(trial, X_train, y_train):
    """Función objetivo optimizada para datos log-transformados"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    
    # Cross-validation más rápida
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=3,  # Reducido a 3-fold para velocidad
        scoring='neg_root_mean_squared_error'
    )
    
    return -cv_scores.mean()

# =============================================
# 3. PIPELINE COMPLETO RÁPIDO
# =============================================

def quick_optimization_pipeline():
    """Pipeline completo optimizado para velocidad"""
    
    # 1. Cargar y transformar
    X, y_log, df, encoders = load_and_transform_data()
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    print(f"\n🔍 OPTIMIZACIÓN RÁPIDA CON OPTUNA")
    print("-" * 40)
    print(f"⚡ Trials: 50 (optimizado para velocidad)")
    print(f"📊 CV: 3-fold (más rápido)")
    
    # 3. Optimización rápida (50 trials)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    
    study.optimize(
        lambda trial: objective_log(trial, X_train, y_train),
        n_trials=50,
        show_progress_bar=True
    )
    
    print(f"\n✅ OPTIMIZACIÓN COMPLETADA")
    print(f"🎯 Mejor RMSE (log): {study.best_value:.4f}")
    
    # 4. Modelo final
    best_model = RandomForestRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # 5. Predicciones
    y_train_pred_log = best_model.predict(X_train)
    y_test_pred_log = best_model.predict(X_test)
    
    # 6. Métricas en escala logarítmica
    train_rmse_log = np.sqrt(mean_squared_error(y_train, y_train_pred_log))
    test_rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred_log))
    train_r2_log = r2_score(y_train, y_train_pred_log)
    test_r2_log = r2_score(y_test, y_test_pred_log)
    
    print(f"\n📊 MÉTRICAS EN ESCALA LOGARÍTMICA:")
    print("-" * 45)
    print(f"RMSE Train (log): {train_rmse_log:.4f}")
    print(f"RMSE Test (log):  {test_rmse_log:.4f}")
    print(f"R² Train:         {train_r2_log:.4f}")
    print(f"R² Test:          {test_r2_log:.4f}")
    
    # 7. CONVERTIR BACK A ESCALA ORIGINAL
    print(f"\n🔄 CONVIRTIENDO A ESCALA ORIGINAL...")
    
    # Convertir predicciones log a escala original
    y_train_pred_original = np.exp(y_train_pred_log)
    y_test_pred_original = np.exp(y_test_pred_log)
    y_train_original = np.exp(y_train)
    y_test_original = np.exp(y_test)
    
    # Métricas en escala original
    train_rmse_orig = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
    train_r2_orig = r2_score(y_train_original, y_train_pred_original)
    test_r2_orig = r2_score(y_test_original, y_test_pred_original)
    
    print(f"\n📊 MÉTRICAS EN ESCALA ORIGINAL:")
    print("-" * 40)
    print(f"RMSE Train: S/ {train_rmse_orig:>12,.0f}")
    print(f"RMSE Test:  S/ {test_rmse_orig:>12,.0f}")
    print(f"R² Train:   {train_r2_orig:>12.4f}")
    print(f"R² Test:    {test_r2_orig:>12.4f}")
    
    # 8. Interpretación
    print(f"\n🎯 EVALUACIÓN DEL RESULTADO:")
    print("-" * 35)
    
    if test_r2_orig > 0.7:
        print("🟢 ¡EXCELENTE! Transformación logarítmica FUNCIONÓ")
        print("✅ R² > 0.7 - Modelo viable")
    elif test_r2_orig > 0.5:
        print("🟡 MODERADO - Transformación ayudó parcialmente")
        print("⚠️ Considerar otras transformaciones")
    elif test_r2_orig > 0:
        print("🟠 LIGERA MEJORA - Pero aún problemático")
        print("💡 Mejor cambiar dataset o problema")
    else:
        print("🔴 TRANSFORMACIÓN NO AYUDÓ")
        print("❌ Confirma que necesitas otro dataset")
    
    # 9. Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 TOP 3 FEATURES MÁS IMPORTANTES:")
    for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # 10. Gráfico simple
    create_simple_plot(y_test_original, y_test_pred_original, test_r2_orig)
    
    return {
        'r2_log': test_r2_log,
        'r2_original': test_r2_orig,
        'rmse_original': test_rmse_orig,
        'study': study,
        'model': best_model,
        'importance': importance_df
    }

def create_simple_plot(y_test, y_pred, r2):
    """Crea gráfico simple de evaluación"""
    
    plt.figure(figsize=(10, 4))
    
    # Predicciones vs Reales
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales (S/)')
    plt.ylabel('Predicciones (S/)')
    plt.title(f'Predicciones vs Reales\nR² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Residuos
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuos (S/)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Residuos')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================
# 4. EJECUCIÓN
# =============================================

if __name__ == "__main__":
    print("🚀 TESTING LOG TRANSFORMATION...")
    results = quick_optimization_pipeline()
    
    print(f"\n🎯 RESULTADO FINAL:")
    print("=" * 30)
    if results['r2_original'] > 0.5:
        print("✅ TRANSFORMACIÓN EXITOSA - CONTINUAR CON ESTE DATASET")
    else:
        print("❌ BUSCAR NUEVO DATASET - ESTE NO ES VIABLE")
    
    print(f"R² Final: {results['r2_original']:.3f}")
    print(f"RMSE Final: S/ {results['rmse_original']:,.0f}")