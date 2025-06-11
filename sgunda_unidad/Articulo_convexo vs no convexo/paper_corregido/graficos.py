import matplotlib.pyplot as plt
import numpy as np

# Configurar matplotlib para texto en español
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# Datos de tu tabla
metodos = ['SVM\nLineal', 'SVM\nRBF', 'Alg.\nGenéticos', 'Reg.\nLogística', 'Redes\nNeuronales', 'Reg.\nRidge']
precision = [0.982, 0.982, 0.982, 0.974, 0.956, 0.956]
f1_score = [0.986, 0.986, 0.986, 0.979, 0.966, 0.966]
auc = [0.994, 0.998, 0.995, 0.996, 0.990, 0.993]

# Posiciones de las barras
x = np.arange(len(metodos))
width = 0.25  # ancho de las barras

# Crear figura única
fig, ax = plt.subplots(figsize=(10, 6))

# Crear barras agrupadas
bars1 = ax.bar(x - width, precision, width, label='Precisión', color='#2E8B57', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, f1_score, width, label='F1-Score', color='#4169E1', alpha=0.8, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, auc, width, label='AUC-ROC', color='#DC143C', alpha=0.8, edgecolor='black', linewidth=0.5)

# Configurar el gráfico
ax.set_xlabel('Métodos de Optimización', fontweight='bold')
ax.set_ylabel('Valores de Métricas', fontweight='bold')
ax.set_title('Comparación de Rendimiento - Técnicas de Optimización\nDiagnóstico de Cáncer de Mama', 
             fontsize=12, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metodos)
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(0.94, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Añadir valores encima de las barras
def agregar_valores(bars, valores):
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

agregar_valores(bars1, precision)
agregar_valores(bars2, f1_score)
agregar_valores(bars3, auc)

# Ajustar layout
plt.tight_layout()

# Guardar la figura en alta resolución
plt.savefig('grafico_resultados_optimizacion.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('grafico_resultados_optimizacion.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()

print("Gráfico guardado como:")
print("- grafico_resultados_optimizacion.png (para insertar en LaTeX)")
print("- grafico_resultados_optimizacion.pdf (versión vectorial)")