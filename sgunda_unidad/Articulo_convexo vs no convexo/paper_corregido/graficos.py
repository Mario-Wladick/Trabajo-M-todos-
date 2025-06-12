import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configurar estilo profesional
plt.style.use('default')

# DATOS EXACTOS DE TU COLAB
data = {
    'Método': ['SVM Lineal', 'SVM RBF', 'Reg. Logística', 'Redes Neuronales', 'Alg. Genéticos', 'Reg. Ridge'],
    'Precisión': [0.9825, 0.9825, 0.9737, 0.9649, 0.9649, 0.9561],
    'F1-Score': [0.9861, 0.9861, 0.9794, 0.9718, 0.9722, 0.9664],
    'AUC-ROC': [0.9937, 0.9897, 0.9957, 0.9940, 0.9947, 0.9927]
}

df = pd.DataFrame(data)

# Crear UNA SOLA figura
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Colores profesionales
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#588157']

# Gráfico de barras agrupadas
x = np.arange(len(df['Método']))
width = 0.25

bars1 = ax.bar(x - width, df['Precisión'], width, label='Precisión', color=colors[0], alpha=0.8)
bars2 = ax.bar(x, df['F1-Score'], width, label='F1-Score', color=colors[1], alpha=0.8)
bars3 = ax.bar(x + width, df['AUC-ROC'], width, label='AUC-ROC', color=colors[2], alpha=0.8)

# Configuración del gráfico
ax.set_ylabel('Rendimiento', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Técnicas de Optimización para Diagnóstico de Cáncer de Mama\nDataset de Wisconsin - Métricas de Rendimiento', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Método'], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0.94, 1.005)

# Agregar valores en las barras
for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    height3 = bar3.get_height()
    
    ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.002,
             f'{height1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.002,
             f'{height2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(bar3.get_x() + bar3.get_width()/2., height3 + 0.002,
             f'{height3:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Destacar empate técnico con línea
empate_y = 0.9825
ax.axhline(y=empate_y, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(1, empate_y + 0.003, 'EMPATE TÉCNICO (98.25%)', ha='center', va='bottom', 
        fontweight='bold', color='red', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))

plt.tight_layout()

# Agregar autor
plt.figtext(0.5, 0.02, 'Autor: Mario W. Ramírez Puma - Universidad Nacional del Altiplano, Puno', 
           ha='center', fontsize=10, style='italic')

# Guardar la figura
plt.savefig('cap2.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figura guardada como 'cap2.png'")
print("🏆 Empate técnico: SVM Lineal y SVM RBF (98.25%)")