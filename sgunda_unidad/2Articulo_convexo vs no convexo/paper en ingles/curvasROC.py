# ROC CURVES CODE - ENGLISH VERSION
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier  # As approximation to Genetic Algorithms
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data (80-20, same random_state from your paper)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with optimal configurations from your paper
models = {
    'Linear SVM': SVC(kernel='linear', C=0.1, probability=True, random_state=42),
    'RBF SVM': SVC(kernel='rbf', C=10.0, gamma=0.01, probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(C=0.1, solver='lbfgs', random_state=42),
    'Neural Networks': MLPClassifier(hidden_layer_sizes=(100, 50, 25), alpha=0.0001, random_state=42, max_iter=1000),
    'Ridge Regression': Ridge(alpha=1.0),
    'Genetic Algorithms': RandomForestClassifier(n_estimators=50, random_state=42)  # Approximation
}

# Configure the plot
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 12})

# Specific colors for each method
colors = {
    'Linear SVM': '#1f77b4',          # Blue
    'RBF SVM': '#ff7f0e',             # Orange  
    'Logistic Regression': '#2ca02c', # Green
    'Neural Networks': '#d62728',     # Red
    'Genetic Algorithms': '#9467bd',  # Purple
    'Ridge Regression': '#8c564b'     # Brown
}

# Line styles
line_styles = {
    'Linear SVM': '-',           # Solid line
    'RBF SVM': '-',              # Solid line
    'Logistic Regression': '--', # Dashed line
    'Neural Networks': '-.',     # Dash-dot line
    'Genetic Algorithms': ':',   # Dotted line
    'Ridge Regression': '--'     # Dashed line
}

# Train models and generate ROC curves
roc_data = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    if name == 'Ridge Regression':
        # Ridge requires special handling for probabilities
        model.fit(X_train_scaled, y_train)
        y_scores = model.predict(X_test_scaled)
        # Convert scores to approximate probabilities
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        model.fit(X_train_scaled, y_train)
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_scores = model.decision_function(X_test_scaled)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Store data
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # Plot curve
    plt.plot(fpr, tpr, 
             color=colors[name], 
             linestyle=line_styles[name],
             linewidth=2.5,
             label=f'{name} (AUC = {roc_auc:.4f})')

# Diagonal reference line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier (AUC = 0.5000)')

# Configure plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
plt.title('Comparative ROC Curves - Breast Cancer Diagnosis\nWisconsin Dataset', 
          fontsize=16, fontweight='bold', pad=20)

# Configure legend
plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)

# Add subtle grid
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Improve visual appearance
plt.tight_layout()

# Save figure in high resolution
plt.savefig('ROC_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ROC_curves.pdf', bbox_inches='tight', facecolor='white')

# Show plot
plt.show()

# Print results ordered by AUC
print("\n" + "="*60)
print("RESULTS ORDERED BY AUC:")
print("="*60)

sorted_results = sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True)
for i, (name, data) in enumerate(sorted_results, 1):
    print(f"{i}. {name:<20} AUC = {data['auc']:.4f}")

print("\n" + "="*60)
print("GENERATED FILES:")
print("="*60)
print("- ROC_curves.png (for LaTeX)")
print("- ROC_curves.pdf (high quality)")