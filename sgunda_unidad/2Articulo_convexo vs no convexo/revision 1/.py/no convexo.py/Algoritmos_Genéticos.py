# Implementación de Algoritmos Genéticos para Dataset de Cáncer de Mama Wisconsin
# Proyecto: Técnicas de Optimización Convexa y No Convexa

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import time
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# =============================================================================

print("="*60)
print("IMPLEMENTACIÓN DE ALGORITMOS GENÉTICOS")
print("Dataset: Wisconsin Breast Cancer")
print("="*60)

# Cargar el dataset
data = load_breast_cancer()
X = data.data  # Características (30 features)
y = data.target  # Etiquetas (0=maligno, 1=benigno)

print(f"\n📊 INFORMACIÓN DEL DATASET:")
print(f"• Número de muestras: {X.shape[0]}")
print(f"• Número de características: {X.shape[1]}")
print(f"• Clases: {data.target_names}")
print(f"• Distribución de clases:")
unique, counts = np.unique(y, return_counts=True)
for i, (clase, count) in enumerate(zip(data.target_names, counts)):
    print(f"  - {clase}: {count} ({count/len(y)*100:.1f}%)")

print(f"\n🧬 CONFIGURACIÓN DE ALGORITMO GENÉTICO:")
print("• Problema: Selección de características + optimización de hiperparámetros")
print("• Cromosoma: [feature_mask(30 bits)] + [C_exp(4 bits)] + [solver(2 bits)]")
print("• Longitud total: 36 bits por individuo")
print("• Función objetivo: F1-Score con validación cruzada")
print("• Operadores: Selección por torneo, cruce uniforme, mutación bit-flip")

# =============================================================================
# 2. PREPROCESAMIENTO DE DATOS
# =============================================================================

print(f"\n🔧 PREPROCESAMIENTO:")

# División entrenamiento/prueba (80/20) - MISMA DIVISIÓN QUE MÉTODOS ANTERIORES
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

# =============================================================================
# 3. IMPLEMENTACIÓN DEL ALGORITMO GENÉTICO
# =============================================================================

class GeneticAlgorithmOptimizer:
    def __init__(self, X_train, y_train, population_size=50, generations=30, 
                 mutation_rate=0.1, crossover_rate=0.8, tournament_size=5):
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.n_features = X_train.shape[1]
        
        # Mapeos para hiperparámetros
        self.C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]  # 2^3 = 8 valores
        self.solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag']  # 2^2 = 4 valores
        
        # Historial de evolución
        self.fitness_history = []
        self.best_fitness_history = []
        
    def create_individual(self):
        """Crear un individuo aleatorio"""
        # 30 bits para features + 3 bits para C + 2 bits para solver
        chromosome = np.random.randint(0, 2, self.n_features + 3 + 2)
        return chromosome
    
    def decode_chromosome(self, chromosome):
        """Decodificar cromosoma a parámetros"""
        # Selección de características (primeros 30 bits)
        feature_mask = chromosome[:self.n_features].astype(bool)
        
        # Si no hay características seleccionadas, seleccionar al menos una
        if not np.any(feature_mask):
            feature_mask[np.random.randint(0, self.n_features)] = True
            
        # Parámetro C (siguientes 3 bits)
        c_bits = chromosome[self.n_features:self.n_features+3]
        c_index = int(''.join(map(str, c_bits)), 2) % len(self.C_values)
        C = self.C_values[c_index]
        
        # Solver (últimos 2 bits)
        solver_bits = chromosome[self.n_features+3:self.n_features+5]
        solver_index = int(''.join(map(str, solver_bits)), 2) % len(self.solvers)
        solver = self.solvers[solver_index]
        
        return feature_mask, C, solver
    
    def evaluate_fitness(self, chromosome):
        """Evaluar fitness de un individuo"""
        try:
            feature_mask, C, solver = self.decode_chromosome(chromosome)
            
            # Seleccionar características
            X_selected = self.X_train[:, feature_mask]
            
            # Crear y evaluar modelo
            model = LogisticRegression(C=C, solver=solver, random_state=42, max_iter=1000)
            
            # Validación cruzada 3-fold (reducida para velocidad)
            scores = cross_val_score(model, X_selected, self.y_train, cv=3, scoring='f1')
            fitness = scores.mean()
            
            # Penalizar por usar demasiadas características (parsimonia)
            n_features_used = np.sum(feature_mask)
            parsimony_penalty = 0.001 * n_features_used / self.n_features
            fitness = fitness - parsimony_penalty
            
            return fitness
        except:
            return 0.0  # Fitness muy bajo para configuraciones inválidas
    
    def tournament_selection(self, population, fitness_scores):
        """Selección por torneo"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected
    
    def uniform_crossover(self, parent1, parent2):
        """Cruce uniforme"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
                
        return child1, child2
    
    def mutate(self, individual):
        """Mutación bit-flip"""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated
    
    def evolve(self):
        """Algoritmo genético principal"""
        print(f"\n🧬 INICIANDO EVOLUCIÓN:")
        print(f"• Población: {self.population_size} individuos")
        print(f"• Generaciones: {self.generations}")
        print(f"• Tasa de mutación: {self.mutation_rate}")
        print(f"• Tasa de cruce: {self.crossover_rate}")
        
        # Inicializar población
        population = [self.create_individual() for _ in range(self.population_size)]
        
        start_time = time.time()
        
        for generation in range(self.generations):
            # Evaluar fitness
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            
            # Estadísticas de la generación
            avg_fitness = np.mean(fitness_scores)
            best_fitness = np.max(fitness_scores)
            
            self.fitness_history.append(avg_fitness)
            self.best_fitness_history.append(best_fitness)
            
            if generation % 5 == 0:
                print(f"• Generación {generation:2d}: Fitness promedio = {avg_fitness:.4f}, Mejor = {best_fitness:.4f}")
            
            # Selección
            selected = self.tournament_selection(population, fitness_scores)
            
            # Cruce y mutación
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                
                child1, child2 = self.uniform_crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        evolution_time = time.time() - start_time
        
        # Encontrar mejor individuo final
        final_fitness_scores = [self.evaluate_fitness(ind) for ind in population]
        best_index = np.argmax(final_fitness_scores)
        best_individual = population[best_index]
        best_fitness = final_fitness_scores[best_index]
        
        print(f"• Evolución completada en {evolution_time:.2f} segundos")
        print(f"• Mejor fitness final: {best_fitness:.4f}")
        
        return best_individual, best_fitness, evolution_time

# =============================================================================
# 4. EJECUCIÓN DEL ALGORITMO GENÉTICO
# =============================================================================

print(f"\n⚙️ OPTIMIZACIÓN CON ALGORITMO GENÉTICO:")

# Crear optimizador genético
ga_optimizer = GeneticAlgorithmOptimizer(
    X_train_scaled, y_train,
    population_size=50,
    generations=30,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=5
)

# Ejecutar evolución
best_chromosome, best_fitness_cv, evolution_time = ga_optimizer.evolve()

# Decodificar mejor solución
best_features, best_C, best_solver = ga_optimizer.decode_chromosome(best_chromosome)

print(f"\n🏆 MEJOR SOLUCIÓN ENCONTRADA:")
print(f"• Características seleccionadas: {np.sum(best_features)} de {len(best_features)}")
print(f"• C óptimo: {best_C}")
print(f"• Solver óptimo: {best_solver}")
print(f"• Fitness (F1-Score CV): {best_fitness_cv:.4f}")

# =============================================================================
# 5. ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

print(f"\n🚀 ENTRENAMIENTO DEL MODELO FINAL:")

# Crear modelo con mejores parámetros
best_model = LogisticRegression(C=best_C, solver=best_solver, random_state=42, max_iter=1000)

# Entrenar con características seleccionadas
X_train_selected = X_train_scaled[:, best_features]
X_test_selected = X_test_scaled[:, best_features]

start_time = time.time()
best_model.fit(X_train_selected, y_train)
training_time = time.time() - start_time

print(f"• Modelo entrenado con características seleccionadas")
print(f"• Tiempo de entrenamiento final: {training_time:.4f} segundos")
print(f"• Tiempo total (evolución + entrenamiento): {evolution_time + training_time:.2f} segundos")

# =============================================================================
# 6. EVALUACIÓN DEL MODELO
# =============================================================================

print(f"\n📊 EVALUACIÓN DEL MODELO:")

# Predicciones
y_pred = best_model.predict(X_test_selected)
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Mostrar resultados
print(f"\n📈 MÉTRICAS DE RENDIMIENTO:")
print(f"• Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"• Precisión (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"• Sensibilidad (Recall): {recall:.4f} ({recall*100:.2f}%)")
print(f"• Puntuación F1: {f1:.4f} ({f1*100:.2f}%)")
print(f"• AUC-ROC: {auc_roc:.4f} ({auc_roc*100:.2f}%)")
print(f"• Tiempo Total: {evolution_time + training_time:.4f} segundos")

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

# Análisis de características seleccionadas
print(f"\n🧬 ANÁLISIS DE EVOLUCIÓN:")
print(f"• Características seleccionadas: {np.sum(best_features)} de {data.feature_names.shape[0]}")
print(f"• Reducción dimensional: {(1 - np.sum(best_features)/len(best_features))*100:.1f}%")

selected_features = data.feature_names[best_features]
print(f"\n🔝 CARACTERÍSTICAS SELECCIONADAS POR ALGORITMO GENÉTICO:")
for i, feature in enumerate(selected_features[:10]):  # Mostrar máximo 10
    print(f"  {i+1:2d}. {feature}")
if len(selected_features) > 10:
    print(f"  ... y {len(selected_features)-10} más")

# Análisis de convergencia
print(f"\n📈 ANÁLISIS DE CONVERGENCIA:")
print(f"• Fitness inicial promedio: {ga_optimizer.fitness_history[0]:.4f}")
print(f"• Fitness final promedio: {ga_optimizer.fitness_history[-1]:.4f}")
print(f"• Mejora total: {ga_optimizer.fitness_history[-1] - ga_optimizer.fitness_history[0]:+.4f}")
print(f"• Mejor fitness en evolución: {max(ga_optimizer.best_fitness_history):.4f}")

# =============================================================================
# 8. COMPARACIÓN CON MÉTODOS ANTERIORES
# =============================================================================

print(f"\n🆚 COMPARACIÓN CON MÉTODOS ANTERIORES:")
print("(Valores de referencia)")
print(f"• SVM Lineal - Accuracy: 0.982")
print(f"• SVM RBF - Accuracy: 0.982")
print(f"• Regresión Logística - Accuracy: 0.974")
print(f"• Redes Neuronales - Accuracy: 0.956")
print(f"• Regresión Ridge - Accuracy: 0.956")
print(f"• Algoritmos Genéticos - Accuracy: {accuracy:.3f}")

ranking_methods = [
    ("SVM Lineal", 0.982), ("SVM RBF", 0.982), ("Regresión Logística", 0.974),
    ("Redes Neuronales", 0.956), ("Regresión Ridge", 0.956), ("Algoritmos Genéticos", accuracy)
]
ranking_methods.sort(key=lambda x: x[1], reverse=True)

print(f"\n🏆 RANKING FINAL:")
for i, (method, acc) in enumerate(ranking_methods):
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}°"
    print(f"  {emoji} {method}: {acc:.3f}")

# =============================================================================
# 9. RESUMEN FINAL PARA EL PAPER
# =============================================================================

print(f"\n" + "="*60)
print("RESUMEN PARA EL PAPER - ALGORITMOS GENÉTICOS")
print("="*60)

print(f"\n📊 RESULTADOS FINALES:")
print(f"• Parámetros evolutivos:")
print(f"  - Población: {ga_optimizer.population_size} individuos")
print(f"  - Generaciones: {ga_optimizer.generations}")
print(f"  - Tasa de mutación: {ga_optimizer.mutation_rate}")
print(f"  - Tasa de cruce: {ga_optimizer.crossover_rate}")
print(f"• Solución óptima:")
print(f"  - Características: {np.sum(best_features)} de {len(best_features)} ({np.sum(best_features)/len(best_features)*100:.1f}%)")
print(f"  - C: {best_C}")
print(f"  - Solver: {best_solver}")
print(f"• Precisión (Accuracy): {accuracy:.3f}")
print(f"• Precisión (Precision): {precision:.3f}")
print(f"• Sensibilidad (Recall): {recall:.3f}")
print(f"• Puntuación F1: {f1:.3f}")
print(f"• AUC-ROC: {auc_roc:.3f}")
print(f"• Tiempo Total: {evolution_time + training_time:.4f}s")

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

print(f"\n✨ VENTAJAS DE ALGORITMOS GENÉTICOS:")
print("• Optimización simultánea de selección de características e hiperparámetros")
print("• No requiere gradientes - optimización libre de derivadas")
print("• Exploración global del espacio de búsqueda")
print("• Capacidad de escape de óptimos locales")

print(f"\n⚠️ CONSIDERACIONES NO CONVEXAS:")
print("• Optimización estocástica - resultados variables entre ejecuciones")
print("• Mayor tiempo computacional debido a evaluaciones iterativas")
print("• Convergencia no garantizada al óptimo global")
print("• Sensible a parámetros evolutivos (población, generaciones, etc.)")

print(f"\n🔬 ASPECTOS EVOLUTIVOS:")
print(f"• Selección automática de características: {(1-np.sum(best_features)/len(best_features))*100:.1f}% reducción")
print(f"• Mejora evolutiva: {max(ga_optimizer.best_fitness_history) - ga_optimizer.fitness_history[0]:+.4f} en F1-Score")
print(f"• Eficiencia temporal: {evolution_time/(ga_optimizer.population_size * ga_optimizer.generations)*1000:.2f}ms por evaluación")

print(f"\n" + "="*60)
print("¡Implementación de Algoritmos Genéticos completada!")
print("="*60)