# PASO 2: REGRESIÓN LINEAL - DATASET AGROINDUSTRIAL ICA
# Predicción de valor_estimado_maximo_venta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. CARGA Y EXPLORACIÓN INICIAL
# =============================================

def load_and_explore_data():
    """Carga el dataset y hace exploración inicial"""
    
    print("🚀 PASO 2: IMPLEMENTACIÓN DE REGRESIÓN LINEAL")
    print("=" * 60)
    
    # Cargar datos desde la carpeta lazypredict_env
    csv_path = r'C:\Users\LENOVO\lazypredict_env\DataSet1_Agroindustria_ICA_2023_rev.csv'
    df = pd.read_csv(csv_path, delimiter=';')
    
    print("\n📊 INFORMACIÓN BÁSICA DEL DATASET")
    print("-" * 40)
    print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    print("\n📋 COLUMNAS DISPONIBLES:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\n💰 ANÁLISIS DE LA VARIABLE OBJETIVO (valor_estimado_maximo_venta)")
    print("-" * 60)
    target = df['valor_estimado_maximo_venta']
    print(f"Media:     S/ {target.mean():>15,.0f}")
    print(f"Mediana:   S/ {target.median():>15,.0f}")
    print(f"Mínimo:    S/ {target.min():>15,.0f}")
    print(f"Máximo:    S/ {target.max():>15,.0f}")
    print(f"Std Dev:   S/ {target.std():>15,.0f}")
    print(f"Coef Var:  {(target.std()/target.mean())*100:>15.1f}%")
    
    print("\n📈 DISTRIBUCIÓN DE VENTAS MÁXIMAS:")
    value_counts = target.value_counts().sort_index()
    for value, count in value_counts.items():
        percentage = (count / len(df)) * 100
        print(f"S/ {value:>12,}: {count:>3} empresas ({percentage:>5.1f}%)")
    
    return df

# =============================================
# 2. PREPARACIÓN DE DATOS PARA REGRESIÓN LINEAL
# =============================================

def prepare_data_for_regression(df):
    """Prepara los datos para regresión lineal"""
    
    print("\n🔧 PREPARACIÓN DE DATOS PARA REGRESIÓN LINEAL")
    print("-" * 50)
    
    # Crear copia para no modificar original
    data = df.copy()
    
    # 1. ENCODING DE VARIABLES CATEGÓRICAS
    print("\n1️⃣ Encoding de variables categóricas:")
    
    categorical_columns = ['provincia', 'distrito', 'descciiu', 'tamanio_emp']
    encoders = {}
    
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        data[f'{col}_encoded'] = encoders[col].fit_transform(data[col])
        n_categories = len(encoders[col].classes_)
        print(f"   • {col}: {n_categories} categorías únicas")
    
    # 2. ENCODING DE VARIABLE BINARIA
    print("\n2️⃣ Encoding de variable binaria:")
    data['exporta_encoded'] = (data['exporta'] == 'SI').astype(int)
    exporta_dist = data['exporta'].value_counts()
    print(f"   • exporta: NO={exporta_dist['NO']}, SI={exporta_dist['SI']}")
    
    # 3. SELECCIÓN DE FEATURES
    print("\n3️⃣ Selección de features para el modelo:")
    feature_columns = [
        'ciiu',                      # Código actividad económica
        'provincia_encoded',         # Provincia (numérico)
        'distrito_encoded',          # Distrito (numérico)
        'descciiu_encoded',          # Tipo de actividad (numérico)
        'tamanio_emp_encoded',       # Tamaño empresa (numérico)
        'exporta_encoded',           # Si exporta (0/1)
        'valor_estimado_minimo_venta' # Valor mínimo venta
    ]
    
    for i, feature in enumerate(feature_columns, 1):
        print(f"   {i}. {feature}")
    
    # Preparar X e y
    X = data[feature_columns]
    y = data['valor_estimado_maximo_venta']
    
    print(f"\n📐 Matriz de features: {X.shape}")
    print(f"📊 Vector objetivo: {y.shape}")
    
    return X, y, encoders, data

# =============================================
# 3. IMPLEMENTACIÓN DE REGRESIÓN LINEAL
# =============================================

def implement_linear_regression(X, y):
    """Implementa y entrena regresión lineal"""
    
    print("\n📈 IMPLEMENTACIÓN DE REGRESIÓN LINEAL")
    print("-" * 45)
    
    # 1. DIVISIÓN TRAIN/TEST
    print("\n1️⃣ División de datos:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   • Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Prueba:        {X_test.shape[0]} muestras")
    
    # 2. ESCALADO DE FEATURES (importante para regresión lineal)
    print("\n2️⃣ Escalado de features:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   • StandardScaler aplicado (media=0, std=1)")
    
    # 3. ENTRENAMIENTO DEL MODELO
    print("\n3️⃣ Entrenamiento del modelo:")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("   • Modelo LinearRegression entrenado exitosamente")
    
    # 4. PREDICCIONES
    print("\n4️⃣ Generando predicciones:")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    print("   • Predicciones para train y test generadas")
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_train_pred, y_test_pred

# =============================================
# 4. EVALUACIÓN DEL MODELO
# =============================================

def evaluate_model(y_train, y_test, y_train_pred, y_test_pred):
    """Evalúa el rendimiento del modelo"""
    
    print("\n📊 EVALUACIÓN DEL MODELO DE REGRESIÓN LINEAL")
    print("-" * 50)
    
    # Calcular métricas
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Mostrar métricas
    print(f"\n{'MÉTRICA':<15} {'ENTRENAMIENTO':<15} {'PRUEBA':<15} {'DIFERENCIA':<12}")
    print("-" * 65)
    print(f"{'RMSE':<15} S/ {train_rmse:<13,.0f} S/ {test_rmse:<13,.0f} {abs(test_rmse-train_rmse):<11,.0f}")
    print(f"{'MAE':<15} S/ {train_mae:<13,.0f} S/ {test_mae:<13,.0f} {abs(test_mae-train_mae):<11,.0f}")
    print(f"{'R²':<15} {train_r2:<15.4f} {test_r2:<15.4f} {abs(test_r2-train_r2):<12.4f}")
    
    # Interpretación de métricas
    print(f"\n🎯 INTERPRETACIÓN DE RESULTADOS:")
    print(f"   • R² = {test_r2:.3f} → El modelo explica {test_r2*100:.1f}% de la variabilidad")
    
    if test_r2 > 0.8:
        print("   • 🟢 Excelente ajuste (R² > 0.8)")
    elif test_r2 > 0.6:
        print("   • 🟡 Buen ajuste (R² > 0.6)")
    elif test_r2 > 0.4:
        print("   • 🟠 Ajuste moderado (R² > 0.4)")
    else:
        print("   • 🔴 Ajuste pobre (R² ≤ 0.4)")
    
    # Análisis de overfitting
    r2_diff = abs(train_r2 - test_r2)
    if r2_diff < 0.05:
        print("   • ✅ Sin overfitting significativo")
    elif r2_diff < 0.1:
        print("   • ⚠️ Ligero overfitting")
    else:
        print("   • ❌ Posible overfitting")
    
    return {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2
    }

# =============================================
# 5. ANÁLISIS DE COEFICIENTES
# =============================================

def analyze_coefficients(model, feature_names):
    """Analiza los coeficientes del modelo lineal"""
    
    print("\n🔍 ANÁLISIS DE COEFICIENTES DEL MODELO")
    print("-" * 45)
    
    # Crear DataFrame con coeficientes
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': model.coef_,
        'Abs_Coeficiente': np.abs(model.coef_)
    }).sort_values('Abs_Coeficiente', ascending=False)
    
    print(f"\n📐 Intercepto: S/ {model.intercept_:,.0f}")
    print(f"\n📊 COEFICIENTES (ordenados por importancia):")
    print("-" * 55)
    print(f"{'FEATURE':<25} {'COEFICIENTE':<15} {'INTERPRETACIÓN'}")
    print("-" * 55)
    
    for _, row in coef_df.iterrows():
        feature = row['Feature']
        coef = row['Coeficiente']
        
        # Interpretación del coeficiente
        if coef > 0:
            direction = "⬆️ Aumenta ventas"
        else:
            direction = "⬇️ Disminuye ventas"
        
        print(f"{feature:<25} {coef:<15,.0f} {direction}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    print("   • Coeficiente positivo = Al aumentar esta variable, aumentan las ventas")
    print("   • Coeficiente negativo = Al aumentar esta variable, disminuyen las ventas")
    print("   • Magnitud = Qué tanto impacto tiene la variable")
    
    return coef_df

# =============================================
# 6. VISUALIZACIONES
# =============================================

def create_visualizations(y_train, y_test, y_train_pred, y_test_pred, coef_df):
    """Crea visualizaciones del análisis"""
    
    print("\n📈 GENERANDO VISUALIZACIONES...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Regresión Lineal - Dataset Agroindustrial', fontsize=16, fontweight='bold')
    
    # 1. Predicciones vs Reales (Test)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Valores Reales (S/)')
    ax1.set_ylabel('Predicciones (S/)')
    ax1.set_title('Predicciones vs Valores Reales (Test)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuos vs Predicciones
    ax2 = axes[0, 1]
    residuals = y_test - y_test_pred
    ax2.scatter(y_test_pred, residuals, alpha=0.6, color='green')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicciones (S/)')
    ax2.set_ylabel('Residuos (S/)')
    ax2.set_title('Residuos vs Predicciones')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de residuos
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Residuos (S/)')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Residuos')
    ax3.grid(True, alpha=0.3)
    
    # 4. Importancia de coeficientes
    ax4 = axes[1, 1]
    top_coef = coef_df.head(6)
    colors = ['red' if x < 0 else 'blue' for x in top_coef['Coeficiente']]
    bars = ax4.barh(range(len(top_coef)), top_coef['Abs_Coeficiente'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(top_coef)))
    ax4.set_yticklabels(top_coef['Feature'])
    ax4.set_xlabel('Valor Absoluto del Coeficiente')
    ax4.set_title('Importancia de Features (Top 6)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================
# 7. FUNCIÓN PRINCIPAL
# =============================================

def main():
    """Función principal que ejecuta todo el pipeline"""
    
    # 1. Cargar y explorar datos
    df = load_and_explore_data()
    
    # 2. Preparar datos
    X, y, encoders, data = prepare_data_for_regression(df)
    
    # 3. Implementar regresión lineal
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_train_pred, y_test_pred = implement_linear_regression(X, y)
    
    # 4. Evaluar modelo
    metrics = evaluate_model(y_train, y_test, y_train_pred, y_test_pred)
    
    # 5. Analizar coeficientes
    coef_df = analyze_coefficients(model, X.columns.tolist())
    
    # 6. Crear visualizaciones
    create_visualizations(y_train, y_test, y_train_pred, y_test_pred, coef_df)
    
    print("\n✅ REGRESIÓN LINEAL COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print(f"📊 R² en Test: {metrics['test_r2']:.3f}")
    print(f"💰 RMSE en Test: S/ {metrics['test_rmse']:,.0f}")
    print("🎯 Modelo baseline establecido para comparar con Random Forest + Optuna")
    
    return model, scaler, metrics, coef_df, encoders

# =============================================
# 8. EJECUCIÓN
# =============================================

if __name__ == "__main__":
    model, scaler, metrics, coef_df, encoders = main()