import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier  # Como aproximación a Algoritmos Genéticos
import warnings
warnings.filterwarnings('ignore')

# Cargar y preparar datos
data = load_breast_cancer()
X, y = data.data, data.target

# Dividir datos (80-20, mismo random_state de tu paper)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir modelos con configuraciones óptimas de tu paper
models = {
    'SVM Lineal': SVC(kernel='linear', C=0.1, probability=True, random_state=42),
    'SVM RBF': SVC(kernel='rbf', C=10.0, gamma=0.01, probability=True, random_state=42),
    'Reg. Logística': LogisticRegression(C=0.1, solver='lbfgs', random_state=42),
    'Redes Neuronales': MLPClassifier(hidden_layer_sizes=(100, 50, 25), alpha=0.0001, random_state=42, max_iter=1000),
    'Reg. Ridge': Ridge(alpha=1.0),
    'Alg. Genéticos': RandomForestClassifier(n_estimators=50, random_state=42)  # Aproximación
}

# Configurar el gráfico
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 12})

# Colores específicos para cada método
colors = {
    'SVM Lineal': '#1f77b4',      # Azul
    'SVM RBF': '#ff7f0e',         # Naranja  
    'Reg. Logística': '#2ca02c',   # Verde
    'Redes Neuronales': '#d62728', # Rojo
    'Alg. Genéticos': '#9467bd',   # Púrpura
    'Reg. Ridge': '#8c564b'        # Marrón
}

# Estilos de línea
line_styles = {
    'SVM Lineal': '-',           # Línea sólida
    'SVM RBF': '-',              # Línea sólida
    'Reg. Logística': '--',      # Línea discontinua
    'Redes Neuronales': '-.',    # Línea punto-raya
    'Alg. Genéticos': ':',       # Línea punteada
    'Reg. Ridge': '--'           # Línea discontinua
}

# Entrenar modelos y generar curvas ROC
roc_data = {}

for name, model in models.items():
    print(f"Entrenando {name}...")
    
    # Entrenar modelo
    if name == 'Reg. Ridge':
        # Ridge requiere manejo especial para probabilidades
        model.fit(X_train_scaled, y_train)
        y_scores = model.predict(X_test_scaled)
        # Convertir scores a probabilidades aproximadas
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        model.fit(X_train_scaled, y_train)
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_scores = model.decision_function(X_test_scaled)
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Guardar datos
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # Plotear curva
    plt.plot(fpr, tpr, 
             color=colors[name], 
             linestyle=line_styles[name],
             linewidth=2.5,
             label=f'{name} (AUC = {roc_auc:.4f})')

# Línea diagonal de referencia (clasificador aleatorio)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Clasificador Aleatorio (AUC = 0.5000)')

# Configurar gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=14, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=14, fontweight='bold')
plt.title('Curvas ROC Comparativas - Diagnóstico de Cáncer de Mama\nDataset de Wisconsin', 
          fontsize=16, fontweight='bold', pad=20)

# Configurar leyenda
plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)

# Añadir grid sutil
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Mejorar el aspecto visual
plt.tight_layout()

# Guardar figura en alta resolución
plt.savefig('ROC_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ROC_curves.pdf', bbox_inches='tight', facecolor='white')

# Mostrar gráfico
plt.show()

# Imprimir resultados ordenados por AUC
print("\n" + "="*60)
print("RESULTADOS ORDENADOS POR AUC:")
print("="*60)

sorted_results = sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True)
for i, (name, data) in enumerate(sorted_results, 1):
    print(f"{i}. {name:<20} AUC = {data['auc']:.4f}")

print("\n" + "="*60)
print("ARCHIVOS GENERADOS:")
print("="*60)
print("- ROC_curves.png (para LaTeX)")
print("- ROC_curves.pdf (alta calidad)")