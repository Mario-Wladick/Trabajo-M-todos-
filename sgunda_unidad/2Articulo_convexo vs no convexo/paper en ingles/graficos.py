
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure professional style
plt.style.use('default')

# EXACT DATA FROM YOUR COLAB
data = {
    'Method': ['Linear SVM', 'RBF SVM', 'Logistic Regression', 'Neural Networks', 'Genetic Algorithms', 'Ridge Regression'],
    'Precision': [0.9825, 0.9825, 0.9737, 0.9649, 0.9649, 0.9561],
    'F1-Score': [0.9861, 0.9861, 0.9794, 0.9718, 0.9722, 0.9664],
    'AUC-ROC': [0.9937, 0.9897, 0.9957, 0.9940, 0.9947, 0.9927]
}

df = pd.DataFrame(data)

# Create ONE SINGLE figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Professional colors
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#588157']

# Grouped bar chart
x = np.arange(len(df['Method']))
width = 0.25

bars1 = ax.bar(x - width, df['Precision'], width, label='Precision', color=colors[0], alpha=0.8)
bars2 = ax.bar(x, df['F1-Score'], width, label='F1-Score', color=colors[1], alpha=0.8)
bars3 = ax.bar(x + width, df['AUC-ROC'], width, label='AUC-ROC', color=colors[2], alpha=0.8)

# Plot configuration
ax.set_ylabel('Performance', fontsize=12, fontweight='bold')
ax.set_title('Comparison of Optimization Techniques for Breast Cancer Diagnosis\nWisconsin Dataset - Performance Metrics', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Method'], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0.94, 1.005)

# Add values on bars
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

# Highlight technical tie with line
tie_y = 0.9825
ax.axhline(y=tie_y, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(1, tie_y + 0.003, 'TECHNICAL TIE (98.25%)', ha='center', va='bottom', 
        fontweight='bold', color='red', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))

plt.tight_layout()

# Add author
plt.figtext(0.5, 0.02, 'Author: Mario W. Ramírez Puma - Universidad Nacional del Altiplano, Puno', 
           ha='center', fontsize=10, style='italic')

# Save the figure
plt.savefig('CAP3.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figure saved as 'CAP3.png'")
print("🏆 Technical tie: Linear SVM and RBF SVM (98.25%)")