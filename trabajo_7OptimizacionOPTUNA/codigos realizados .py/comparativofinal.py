# PASO 5: ANÁLISIS COMPARATIVO FINAL
# Consolidación de resultados y conclusiones del proyecto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo profesional
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================
# 1. CONSOLIDACIÓN DE RESULTADOS
# =============================================

def consolidate_results():
    """Consolida todos los resultados obtenidos"""
    
    print("📊 PASO 5: ANÁLISIS COMPARATIVO FINAL")
    print("=" * 55)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"🎯 Proyecto: Métodos de Optimización en Regresión")
    print(f"📈 Dataset: Empresas Agroindustriales de Ica (494 empresas)")
    
    # Resultados de todos los métodos probados
    results_summary = {
        'Método': [
            'Regresión Lineal (Baseline)',
            'Random Forest (Sin transformar)',
            'Random Forest + Log Transform + Optuna'
        ],
        'R² Train': [
            0.8980,      # Regresión lineal
            0.0632,      # Random Forest original
            0.9763       # Random Forest + Log
        ],
        'R² Test': [
            -16737.2072, # Regresión lineal  
            -1605.5209,  # Random Forest original
            0.9987       # Random Forest + Log
        ],
        'RMSE Test (S/)': [
            195_455_534, # Regresión lineal
            60_553_142,  # Random Forest original
            54_013       # Random Forest + Log
        ],
        'Tiempo Entrenamiento': [
            '<1 segundo',
            '~50 segundos',
            '~180 segundos'
        ],
        'Viable': [
            '❌ No',
            '❌ No', 
            '✅ Sí'
        ]
    }
    
    df_results = pd.DataFrame(results_summary)
    
    print(f"\n📋 RESUMEN COMPARATIVO DE MÉTODOS")
    print("-" * 80)
    print(df_results.to_string(index=False))
    
    return df_results

# =============================================
# 2. ANÁLISIS DE TRANSFORMACIÓN DE DATOS
# =============================================

def analyze_data_transformation():
    """Analiza el impacto de la transformación logarítmica"""
    
    print(f"\n🔄 ANÁLISIS DE TRANSFORMACIÓN DE DATOS")
    print("-" * 50)
    
    # Datos de transformación (obtenidos anteriormente)
    transformation_impact = {
        'Métrica': ['Mínimo', 'Máximo', 'Media', 'Std Dev', 'Coef. Variación'],
        'Datos Originales': [
            'S/ 742,500',
            'S/ 13,365,000,000', 
            'S/ 29,340,774',
            'S/ 601,274,319',
            '2,049.3%'
        ],
        'Datos Log-Transformados': [
            '13.52',
            '23.32',
            '13.75', 
            '0.91',
            '6.6%'
        ],
        'Mejora': [
            '-',
            '-',
            '-',
            '660,000x menor',
            '310x mejor'
        ]
    }
    
    df_transform = pd.DataFrame(transformation_impact)
    print(df_transform.to_string(index=False))
    
    print(f"\n💡 CONCLUSIONES DE LA TRANSFORMACIÓN:")
    print("   • Transformación logarítmica fue CLAVE para el éxito")
    print("   • Redujo variabilidad extrema 310 veces") 
    print("   • Permitió que Random Forest encontrara patrones")
    print("   • Sin transformación: R² negativo")
    print("   • Con transformación: R² = 99.87%")

# =============================================
# 3. ANÁLISIS DE OPTUNA
# =============================================

def analyze_optuna_optimization():
    """Analiza el proceso de optimización con Optuna"""
    
    print(f"\n🔍 ANÁLISIS DE OPTIMIZACIÓN CON OPTUNA")
    print("-" * 45)
    
    # Hiperparámetros optimizados (del resultado anterior)
    optuna_results = {
        'Hiperparámetro': [
            'n_estimators',
            'max_depth', 
            'min_samples_split',
            'min_samples_leaf',
            'max_features',
            'bootstrap'
        ],
        'Valor Óptimo': [
            '150',
            '16',
            '16', 
            '10',
            'sqrt',
            'True'
        ],
        'Rango Explorado': [
            '50-300',
            '5-20',
            '2-15',
            '1-8', 
            'sqrt, log2',
            'True, False'
        ],
        'Impacto': [
            'Alto',
            'Medio',
            'Medio',
            'Bajo',
            'Alto', 
            'Medio'
        ]
    }
    
    df_optuna = pd.DataFrame(optuna_results)
    print(df_optuna.to_string(index=False))
    
    print(f"\n🎯 BENEFICIOS DE OPTUNA:")
    print("   • Exploró 50 combinaciones de hiperparámetros automáticamente")
    print("   • Encontró configuración óptima en ~3 minutos")
    print("   • TPE Sampler: enfoque inteligente, no fuerza bruta")
    print("   • Cross-validation: aseguró robustez del modelo")
    print("   • Mejor RMSE: 0.2898 en escala logarítmica")

# =============================================
# 4. ANÁLISIS DE FEATURES IMPORTANTES
# =============================================

def analyze_feature_importance():
    """Analiza las variables más importantes del modelo final"""
    
    print(f"\n🔍 VARIABLES MÁS IMPORTANTES (MODELO FINAL)")
    print("-" * 55)
    
    # Feature importance del modelo final
    features_importance = {
        'Variable': [
            'valor_estimado_minimo_venta',
            'tamanio_emp_encoded', 
            'exporta_encoded',
            'ciiu',
            'descciiu_encoded',
            'distrito_encoded',
            'provincia_encoded'
        ],
        'Importancia': [
            0.537,
            0.288,
            0.113,
            0.035,
            0.018,
            0.007,
            0.002
        ],
        'Porcentaje': [
            '53.7%',
            '28.8%', 
            '11.3%',
            '3.5%',
            '1.8%',
            '0.7%',
            '0.2%'
        ],
        'Interpretación Empresarial': [
            'Ventas mínimas predicen ventas máximas',
            'Tamaño de empresa es factor clave',
            'Actividad exportadora es relevante', 
            'Código de actividad económica',
            'Tipo específico de actividad',
            'Ubicación distrital menor impacto',
            'Provincia tiene mínimo impacto'
        ]
    }
    
    df_features = pd.DataFrame(features_importance)
    print(df_features.to_string(index=False))
    
    print(f"\n💼 CONCLUSION:")
    print("   • TOP 3 variables explican 79.8% de la predicción")
    print("   • Ventas mínimas son el mejor predictor (lógico empresarialmente)")
    print("   • Tamaño empresarial define capacidad de ventas") 
    print("   • Exportar indica empresa más sofisticada")
    print("   • Ubicación geográfica tiene impacto mínimo")

# =============================================
# 5. VISUALIZACIÓN COMPARATIVA
# =============================================

def create_comparative_visualization():
    """Crea visualización comparativa de todos los métodos"""
    
    print(f"\n📈 GENERANDO VISUALIZACIÓN COMPARATIVA...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Comparación R² 
    ax1 = axes[0, 0]
    methods = ['Reg. Lineal', 'RF Original', 'RF + Log + Optuna']
    r2_values = [-16737.2072, -1605.5209, 0.9987]
    colors = ['red', 'orange', 'green']
    
    # Usar escala logarítmica para mostrar la mejora
    ax1.bar(methods, [abs(x) if x < 0 else x for x in r2_values], color=colors, alpha=0.7)
    ax1.set_ylabel('|R²| (Escala log)')
    ax1.set_title('📊 Comparación R² por Método', fontweight='bold', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, v in enumerate(r2_values):
        if v > 0:
            ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(i, abs(v), f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Evolución del RMSE
    ax2 = axes[0, 1] 
    rmse_values = [195_455_534, 60_553_142, 54_013]
    ax2.plot(methods, rmse_values, 'o-', linewidth=3, markersize=8, color='blue')
    ax2.set_ylabel('RMSE (S/)')
    ax2.set_title('📉 Evolución del RMSE', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Formato de números
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x/1e6:.0f}M'))
    
    # 3. Feature Importance (Top 5)
    ax3 = axes[1, 0]
    features = ['valor_min_venta', 'tamanio_emp', 'exporta', 'ciiu', 'descciiu']
    importance = [0.537, 0.288, 0.113, 0.035, 0.018]
    
    bars = ax3.barh(features, importance, color='skyblue', alpha=0.8)
    ax3.set_xlabel('Importancia')
    ax3.set_title('🔍 Top 5 Variables Más Importantes', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Añadir porcentajes
    for i, v in enumerate(importance):
        ax3.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontweight='bold')
    
    # 4. Impacto de la transformación
    ax4 = axes[1, 1]
    transform_metrics = ['Coef. Variación', 'R² Final', 'RMSE Final']
    before = [2049.3, -1605.5, 60_553_142/1e6]  # Valores antes
    after = [6.6, 0.9987, 54_013/1e6]  # Valores después
    
    x = np.arange(len(transform_metrics))
    width = 0.35
    
    ax4.bar(x - width/2, [2049.3, 0, 60.6], width, label='Sin Transformación', 
            color='red', alpha=0.7)
    ax4.bar(x + width/2, [6.6, 99.87, 0.054], width, label='Con Log Transform', 
            color='green', alpha=0.7)
    
    ax4.set_ylabel('Valor de Métrica')
    ax4.set_title('🔄 Impacto de Transformación Log', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['CV (%)', 'R² (%)', 'RMSE (M S/)'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================
# 6. CONCLUSIONES Y RECOMENDACIONES
# =============================================

def generate_conclusions():
    """Genera conclusiones finales del proyecto"""
    
    print(f"\n🎯 RESUMEN")
    print("=" * 30)
    print("   • Transformación logarítmica resolvió problema de outliers")
    print("   • Random Forest + Optuna logró R² = 99.87%")
    print("   • Proyecto técnicamente exitoso")

# =============================================
# 7. FUNCIÓN PRINCIPAL
# =============================================

def main():
    """Ejecuta análisis comparativo completo"""
    
    # 1. Consolidar resultados
    df_results = consolidate_results()
    
    # 2. Analizar transformación
    analyze_data_transformation()
    
    # 3. Analizar Optuna
    analyze_optuna_optimization()
    
    # 4. Analizar features
    analyze_feature_importance()
    
    # 5. Crear visualizaciones
    create_comparative_visualization()
    
    # 6. Conclusiones finales
    generate_conclusions()
    
    print(f"\n📊 ANÁLISIS TÉCNICO COMPLETADO")
  
    
    return df_results

# =============================================
# 8. EJECUCIÓN
# =============================================

if __name__ == "__main__":
    results_df = main()