# PASO 4: RANDOM FOREST + OPTUNA - DATASET AGROINDUSTRIAL ICA
# Predicción de valor_estimado_maximo_venta con optimización de hiperparámetros

import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================

def load_and_prepare_data():
    """Carga y prepara el dataset para Random Forest"""
    
    print("🌳 PASO 4: RANDOM FOREST + OPTUNA OPTIMIZATION")
    print("=" * 65)
    
    # Cargar datos desde la carpeta lazypredict_env
    csv_path = r'C:\Users\LENOVO\lazypredict_env\DataSet1_Agroindustria_ICA_2023_rev.csv'
    df = pd.read_csv(csv_path, delimiter=';')
    
    print("\n📊 INFORMACIÓN DEL DATASET")
    print("-" * 40)
    print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Empresas por tamaño: {dict(df['tamanio_emp'].value_counts())}")
    print(f"Empresas exportadoras: {sum(df['exporta'] == 'SI')} de {len(df)}")
    
    # Análisis de variable objetivo
    print(f"\n💰 VARIABLE OBJETIVO: valor_estimado_maximo_venta")
    print("-" * 50)
    target = df['valor_estimado_maximo_venta']
    print(f"Media:      S/ {target.mean():>15,.0f}")
    print(f"Mediana:    S/ {target.median():>15,.0f}")
    print(f"Min:        S/ {target.min():>15,.0f}")
    print(f"Max:        S/ {target.max():>15,.0f}")
    print(f"Q1:         S/ {target.quantile(0.25):>15,.0f}")
    print(f"Q3:         S/ {target.quantile(0.75):>15,.0f}")
    
    # Preparar encoders para variables categóricas
    print(f"\n🔧 PREPARACIÓN DE FEATURES")
    print("-" * 35)
    
    # Encoding de variables categóricas
    categorical_features = ['provincia', 'distrito', 'descciiu', 'tamanio_emp']
    encoders = {}
    
    for feature in categorical_features:
        encoders[feature] = LabelEncoder()
        df[f'{feature}_encoded'] = encoders[feature].fit_transform(df[feature])
        n_unique = df[feature].nunique()
        print(f"✓ {feature}: {n_unique} categorías → encoded")
    
    # Variable binaria exporta
    df['exporta_encoded'] = (df['exporta'] == 'SI').astype(int)
    print(f"✓ exporta: binaria → {sum(df['exporta'] == 'SI')} exportadoras")
    
    # Seleccionar features finales para Random Forest
    feature_columns = [
        'ciiu',                      # Código actividad económica
        'provincia_encoded',         # Provincia (0-4)
        'distrito_encoded',          # Distrito (0-33)
        'descciiu_encoded',          # Actividad detallada (0-13)
        'tamanio_emp_encoded',       # Tamaño empresa (0-3)
        'exporta_encoded',           # Exporta (0/1)
        'valor_estimado_minimo_venta' # Ventas mínimas
    ]
    
    X = df[feature_columns]
    y = df['valor_estimado_maximo_venta']
    
    print(f"\n📐 FEATURES SELECCIONADAS: {len(feature_columns)}")
    for i, feature in enumerate(feature_columns, 1):
        print(f"  {i}. {feature}")
    
    print(f"\n📊 DIMENSIONES FINALES:")
    print(f"  • X (features): {X.shape}")
    print(f"  • y (target):   {y.shape}")
    
    return X, y, encoders, df

# =============================================
# 2. FUNCIÓN OBJETIVO PARA OPTUNA
# =============================================

def objective(trial, X_train, y_train):
    """
    Función objetivo para Optuna
    Optimiza hiperparámetros de Random Forest
    Retorna RMSE promedio de validación cruzada
    """
    
    # Hiperparámetros a optimizar
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1  # Usar todos los cores disponibles
    }
    
    # Crear modelo con hiperparámetros sugeridos
    model = RandomForestRegressor(**params)
    
    # Validación cruzada 5-fold
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    # Retornar RMSE promedio (Optuna minimiza)
    rmse = -cv_scores.mean()
    
    return rmse

# =============================================
# 3. OPTIMIZACIÓN CON OPTUNA
# =============================================

def optimize_with_optuna(X_train, y_train, n_trials=100):
    """Optimiza hiperparámetros usando Optuna"""
    
    print(f"\n🔍 OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    print("-" * 55)
    print(f"🎯 Objetivo: Minimizar RMSE")
    print(f"🔄 Trials: {n_trials}")
    print(f"📊 Validación: 5-fold cross-validation")
    print(f"⏱️ Sampler: TPE (Tree-structured Parzen Estimator)")
    
    # Crear estudio Optuna
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='RandomForest_Optimization'
    )
    
    print(f"\n🚀 INICIANDO OPTIMIZACIÓN...")
    print("⏳ Esto puede tomar varios minutos...")
    
    # Optimizar con callback para mostrar progreso cada 20 trials
    def progress_callback(study, trial):
        if trial.number % 20 == 0:
            print(f"Trial {trial.number:3d}: RMSE = {trial.value:>12,.0f}")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        callbacks=[progress_callback]
    )
    
    print(f"\n✅ OPTIMIZACIÓN COMPLETADA")
    print("-" * 40)
    print(f"📊 Trials ejecutados: {len(study.trials)}")
    print(f"🎯 Mejor RMSE: S/ {study.best_value:,.0f}")
    print(f"⏱️ Tiempo total: {sum([t.duration.total_seconds() for t in study.trials if t.duration]):.1f}s")
    
    print(f"\n🏆 MEJORES HIPERPARÁMETROS:")
    print("-" * 35)
    for param, value in study.best_params.items():
        print(f"  • {param:<18}: {value}")
    
    return study

# =============================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# =============================================

def train_final_model(X_train, X_test, y_train, y_test, best_params):
    """Entrena modelo final con mejores hiperparámetros"""
    
    print(f"\n🎯 ENTRENAMIENTO DEL MODELO FINAL")
    print("-" * 45)
    
    # Crear modelo con mejores parámetros
    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    
    print(f"🌳 Entrenando Random Forest con hiperparámetros optimizados...")
    final_model.fit(X_train, y_train)
    
    # Generar predicciones
    print(f"📊 Generando predicciones...")
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    
    # Calcular métricas
    metrics = calculate_metrics(y_train, y_test, y_train_pred, y_test_pred)
    
    print(f"\n📈 MÉTRICAS DEL MODELO FINAL")
    print("-" * 40)
    print(f"{'MÉTRICA':<12} {'TRAIN':<15} {'TEST':<15} {'DIFERENCIA':<12}")
    print("-" * 60)
    print(f"{'RMSE':<12} S/ {metrics['train_rmse']:<13,.0f} S/ {metrics['test_rmse']:<13,.0f} {abs(metrics['test_rmse']-metrics['train_rmse']):<11,.0f}")
    print(f"{'MAE':<12} S/ {metrics['train_mae']:<13,.0f} S/ {metrics['test_mae']:<13,.0f} {abs(metrics['test_mae']-metrics['train_mae']):<11,.0f}")
    print(f"{'R²':<12} {metrics['train_r2']:<15.4f} {metrics['test_r2']:<15.4f} {abs(metrics['test_r2']-metrics['train_r2']):<12.4f}")
    
    # Interpretación de resultados
    interpret_results(metrics)
    
    return final_model, y_test_pred, metrics

def calculate_metrics(y_train, y_test, y_train_pred, y_test_pred):
    """Calcula todas las métricas de evaluación"""
    return {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }

def interpret_results(metrics):
    """Interpreta los resultados del modelo"""
    
    test_r2 = metrics['test_r2']
    r2_diff = abs(metrics['train_r2'] - metrics['test_r2'])
    
    print(f"\n🎯 INTERPRETACIÓN DE RESULTADOS:")
    print("-" * 40)
    
    # Interpretación de R²
    if test_r2 > 0.9:
        print(f"📊 R² = {test_r2:.3f} → 🟢 EXCELENTE AJUSTE (>90%)")
    elif test_r2 > 0.8:
        print(f"📊 R² = {test_r2:.3f} → 🟢 MUY BUEN AJUSTE (>80%)")
    elif test_r2 > 0.7:
        print(f"📊 R² = {test_r2:.3f} → 🟡 BUEN AJUSTE (>70%)")
    elif test_r2 > 0.5:
        print(f"📊 R² = {test_r2:.3f} → 🟠 AJUSTE MODERADO (>50%)")
    else:
        print(f"📊 R² = {test_r2:.3f} → 🔴 AJUSTE POBRE (<50%)")
    
    # Análisis de overfitting
    if r2_diff < 0.05:
        print(f"🎯 Diferencia R²: {r2_diff:.3f} → ✅ SIN OVERFITTING")
    elif r2_diff < 0.1:
        print(f"🎯 Diferencia R²: {r2_diff:.3f} → ⚠️ LIGERO OVERFITTING")
    else:
        print(f"🎯 Diferencia R²: {r2_diff:.3f} → ❌ POSIBLE OVERFITTING")

# =============================================
# 5. ANÁLISIS DE IMPORTANCIA DE FEATURES
# =============================================

def analyze_feature_importance(model, feature_names):
    """Analiza la importancia de las features"""
    
    print(f"\n🔍 IMPORTANCIA DE FEATURES")
    print("-" * 35)
    
    # Crear DataFrame con importancias
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calcular porcentajes
    importance_df['percentage'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
    
    print(f"{'FEATURE':<25} {'IMPORTANCIA':<12} {'PORCENTAJE':<10}")
    print("-" * 50)
    
    for _, row in importance_df.iterrows():
        feature = row['feature']
        importance = row['importance']
        percentage = row['percentage']
        print(f"{feature:<25} {importance:<12.4f} {percentage:<10.1f}%")
    
    print(f"\n💡 INTERPRETACIÓN:")
    print(f"   • Mayor importancia = Mayor impacto en predicción")
    print(f"   • Top 3 features explican: {importance_df.head(3)['percentage'].sum():.1f}%")
    
    return importance_df

# =============================================
# 6. VISUALIZACIONES COMPLETAS
# =============================================

def create_comprehensive_visualizations(study, model, y_test, y_test_pred, importance_df, metrics):
    """Crea visualizaciones completas del análisis"""
    
    print(f"\n📈 GENERANDO VISUALIZACIONES...")
    
    # Configurar subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Historia de optimización Optuna
    ax1 = plt.subplot(2, 3, 1)
    trials_df = study.trials_dataframe()
    ax1.plot(trials_df['number'], trials_df['value'], alpha=0.7, linewidth=1)
    ax1.set_title('📊 Historia de Optimización Optuna', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    
    # Línea del mejor resultado
    best_trial = trials_df.loc[trials_df['value'].idxmin()]
    ax1.axhline(y=best_trial['value'], color='red', linestyle='--', alpha=0.8, 
                label=f'Mejor: {best_trial["value"]:,.0f}')
    ax1.legend()
    
    # 2. Predicciones vs Reales
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Valores Reales (S/)')
    ax2.set_ylabel('Predicciones (S/)')
    ax2.set_title(f'🎯 Predicciones vs Reales (R² = {metrics["test_r2"]:.3f})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de residuos
    ax3 = plt.subplot(2, 3, 3)
    residuals = y_test - y_test_pred
    ax3.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Residuos (S/)')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('📊 Distribución de Residuos', fontsize=14, fontweight='bold')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Importancia de features
    ax4 = plt.subplot(2, 3, 4)
    top_features = importance_df.head(6)
    bars = ax4.barh(range(len(top_features)), top_features['importance'])
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['feature'])
    ax4.set_xlabel('Importancia')
    ax4.set_title('🔍 Importancia de Features (Top 6)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Importancia de hiperparámetros (Optuna)
    ax5 = plt.subplot(2, 3, 5)
    param_importance = optuna.importance.get_param_importances(study)
    params = list(param_importance.keys())[:6]  # Top 6
    importances = [param_importance[p] for p in params]
    
    bars = ax5.bar(range(len(params)), importances)
    ax5.set_xticks(range(len(params)))
    ax5.set_xticklabels(params, rotation=45, ha='right')
    ax5.set_ylabel('Importancia')
    ax5.set_title('🔧 Importancia de Hiperparámetros', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Métricas comparativas (CORREGIDO)
    ax6 = plt.subplot(2, 3, 6)
    
    # Separar métricas de error y R²
    error_metrics = ['RMSE', 'MAE']
    train_errors = [metrics['train_rmse']/1e6, metrics['train_mae']/1e6]
    test_errors = [metrics['test_rmse']/1e6, metrics['test_mae']/1e6]
    
    x = np.arange(len(error_metrics))
    width = 0.35
    
    # Gráfico de barras para RMSE y MAE
    ax6.bar(x - width/2, train_errors, width, label='Train', alpha=0.8, color='skyblue')
    ax6.bar(x + width/2, test_errors, width, label='Test', alpha=0.8, color='orange')
    
    ax6.set_xlabel('Métricas de Error')
    ax6.set_ylabel('RMSE, MAE (Millones S/)')
    ax6.set_title('📊 Comparación Train vs Test', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(error_metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Añadir texto con R²
    r2_text = f"R² Train: {metrics['train_r2']:.3f}\nR² Test: {metrics['test_r2']:.3f}"
    ax6.text(0.02, 0.98, r2_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# =============================================
# 7. FUNCIÓN PRINCIPAL
# =============================================

def main():
    """Función principal que ejecuta todo el pipeline"""
    
    # 1. Cargar y preparar datos
    X, y, encoders, df = load_and_prepare_data()
    
    # 2. Dividir en train/test
    print(f"\n📊 DIVISIÓN DE DATOS")
    print("-" * 25)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Prueba:        {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # 3. Optimización con Optuna
    study = optimize_with_optuna(X_train, y_train, n_trials=100)
    
    # 4. Entrenar modelo final
    final_model, y_test_pred, metrics = train_final_model(
        X_train, X_test, y_train, y_test, study.best_params
    )
    
    # 5. Análisis de feature importance
    importance_df = analyze_feature_importance(final_model, X.columns.tolist())
    
    # 6. Crear visualizaciones
    create_comprehensive_visualizations(study, final_model, y_test, y_test_pred, 
                                      importance_df, metrics)
    
    # 7. Resumen final
    print(f"\n🎉 RANDOM FOREST + OPTUNA COMPLETADO")
    print("=" * 50)
    print(f"🎯 R² Final: {metrics['test_r2']:.3f}")
    print(f"💰 RMSE Final: S/ {metrics['test_rmse']:,.0f}")
    print(f"🔧 Trials Optuna: {len(study.trials)}")
    print(f"🌳 Mejor n_estimators: {study.best_params['n_estimators']}")
    print(f"📊 Feature más importante: {importance_df.iloc[0]['feature']}")
    
    return study, final_model, metrics, importance_df, encoders

# =============================================
# 8. EJECUCIÓN
# =============================================

if __name__ == "__main__":
    study, model, metrics, importance, encoders = main()
    
    # Opcional: Guardar resultados
    print(f"\n💾 GUARDANDO RESULTADOS...")
    study.trials_dataframe().to_csv('optuna_trials_results.csv', index=False)
    importance.to_csv('feature_importance_results.csv', index=False)
    print(f"✓ Resultados guardados en CSV")