from manim import *
import numpy as np

class PaperCompletoSinErrores(Scene):
    def construct(self):
        # 1. INTRODUCCIÓN EXTENDIDA
        self.introduccion_extendida()
        self.clear()
        self.wait(1)
        
        # 2. DATASET WISCONSIN DETALLADO
        self.dataset_wisconsin_detallado()
        self.clear()
        self.wait(1)
        
        # 3. METODOLOGÍA COMPLETA
        self.metodologia_completa()
        self.clear()
        self.wait(1)
        
        # 4. PREPROCESAMIENTO DETALLADO
        self.preprocesamiento_detallado()
        self.clear()
        self.wait(1)
        
        # 5. CONFIGURACIONES ÓPTIMAS
        self.configuraciones_optimas()
        self.clear()
        self.wait(1)
        
        # 6. RESULTADOS EXPERIMENTALES
        self.resultados_experimentales()
        self.clear()
        self.wait(1)
        
        # 7. ANÁLISIS DE MATRICES
        self.analisis_matrices_completo()
        self.clear()
        self.wait(1)
        
        # 8. CURVAS ROC COMPLETAS
        self.curvas_roc_completas()
        self.clear()
        self.wait(1)
        
        # 9. DISCUSIÓN PROFUNDA
        self.discusion_profunda()
        self.clear()
        self.wait(1)
        
        # 10. CONCLUSIONES FINALES
        self.conclusiones_finales()
    
    def introduccion_extendida(self):
        """Introducción completa de 60 segundos"""
        # Título dividido para evitar solapamiento
        titulo_linea1 = Text("Analisis Comparativo en Tecnicas de Optimizacion", 
                            font_size=24, color=BLUE)
        titulo_linea2 = Text("Convexa y No Convexa para Diagnostico", 
                            font_size=20, color=BLUE)
        titulo_linea3 = Text("de Cancer de Mama", 
                            font_size=20, color=BLUE)
        
        titulo_linea1.to_edge(UP, buff=0.5)
        titulo_linea2.next_to(titulo_linea1, DOWN, buff=0.2)
        titulo_linea3.next_to(titulo_linea2, DOWN, buff=0.2)
        
        # Autor y afiliación
        autor = Text("Mario Wilfredo Ramirez Puma", font_size=16, color=YELLOW)
        universidad = Text("Universidad Nacional del Altiplano - Puno", 
                          font_size=12, color=GRAY)
        escuela = Text("Escuela Profesional de Ingenieria Estadistica e Informatica", 
                      font_size=10, color=GRAY)
        
        autor.next_to(titulo_linea3, DOWN, buff=0.4)
        universidad.next_to(autor, DOWN, buff=0.2)
        escuela.next_to(universidad, DOWN, buff=0.1)
        
        self.play(Write(titulo_linea1))
        self.wait(0.8)
        self.play(Write(titulo_linea2))
        self.play(Write(titulo_linea3))
        self.wait(1)
        self.play(Write(autor))
        self.play(Write(universidad))
        self.play(Write(escuela))
        self.wait(2)
        
        # Limpiar y mostrar problema
        self.play(FadeOut(universidad, escuela))
        
        problema_titulo = Text("PROBLEMA DE INVESTIGACION", font_size=22, color=RED)
        problema_titulo.move_to(UP * 1.5)
        
        pregunta_linea1 = Text("Cuando se justifica el uso de metodos", 
                              font_size=16, color=WHITE)
        pregunta_linea2 = Text("no convexos sobre convexos", 
                              font_size=16, color=WHITE)
        pregunta_linea3 = Text("en diagnostico medico?", 
                              font_size=16, color=WHITE)
        
        pregunta_linea1.next_to(problema_titulo, DOWN, buff=0.6)
        pregunta_linea2.next_to(pregunta_linea1, DOWN, buff=0.2)
        pregunta_linea3.next_to(pregunta_linea2, DOWN, buff=0.2)
        
        # Motivación
        motivacion_titulo = Text("MOTIVACION:", font_size=18, color=YELLOW)
        motivacion_titulo.next_to(pregunta_linea3, DOWN, buff=0.8)
        
        motivacion_items = VGroup(
            Text("El diagnostico temprano SALVA VIDAS", font_size=14, color=GREEN),
            Text("La precision algoritmica impacta supervivencia", font_size=14, color=YELLOW),
            Text("Balance entre precision, eficiencia e interpretabilidad", font_size=14, color=BLUE)
        )
        motivacion_items.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        motivacion_items.next_to(motivacion_titulo, DOWN, buff=0.4)
        
        self.play(Write(problema_titulo))
        self.wait(1)
        self.play(Write(pregunta_linea1))
        self.play(Write(pregunta_linea2))
        self.play(Write(pregunta_linea3))
        self.wait(2)
        
        self.play(Write(motivacion_titulo))
        self.wait(1)
        for item in motivacion_items:
            self.play(Write(item))
            self.wait(1)
        
        self.wait(3)
    
    def dataset_wisconsin_detallado(self):
        """Dataset Wisconsin detallado - 45 segundos"""
        titulo = Text("DATASET DE WISCONSIN", font_size=28, color=BLUE)
        titulo.to_edge(UP)
        
        # Descripción técnica
        descripcion_titulo = Text("CARACTERISTICAS DE IMAGENES FNA", font_size=18, color=YELLOW)
        descripcion_titulo.next_to(titulo, DOWN, buff=0.6)
        
        descripcion_texto = Text("Fine Needle Aspiration - Aspirados de aguja fina", 
                                font_size=14, color=GRAY)
        descripcion_texto.next_to(descripcion_titulo, DOWN, buff=0.3)
        
        # Estadísticas principales
        stats_titulo = Text("ESTADISTICAS DEL DATASET", font_size=18, color=ORANGE)
        stats_titulo.next_to(descripcion_texto, DOWN, buff=0.8)
        
        stats_items = VGroup(
            Text("Total de muestras: 569 casos", font_size=16, color=WHITE),
            Text("Casos benignos: 357 (62.7%)", font_size=16, color=GREEN),
            Text("Casos malignos: 212 (37.3%)", font_size=16, color=RED),
            Text("Caracteristicas: 30 atributos numericos", font_size=16, color=BLUE)
        )
        stats_items.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        stats_items.next_to(stats_titulo, DOWN, buff=0.4)
        
        # División de datos explicada
        division_titulo = Text("DIVISION DE DATOS", font_size=16, color=PURPLE)
        division_titulo.next_to(stats_items, DOWN, buff=0.8)
        
        division_info = VGroup(
            Text("80% Entrenamiento: 455 muestras", font_size=14, color=GREEN),
            Text("20% Prueba: 114 muestras", font_size=14, color=BLUE)
        )
        division_info.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        division_info.next_to(division_titulo, DOWN, buff=0.3)
        
        # Animaciones
        self.play(Write(titulo))
        self.wait(1)
        
        self.play(Write(descripcion_titulo))
        self.play(Write(descripcion_texto))
        self.wait(2)
        
        self.play(Write(stats_titulo))
        self.wait(1)
        for item in stats_items:
            self.play(Write(item))
            self.wait(0.8)
        
        self.play(Write(division_titulo))
        self.wait(1)
        for item in division_info:
            self.play(Write(item))
            self.wait(0.8)
        
        self.wait(3)
    
    def metodologia_completa(self):
        """Metodología completa - 80 segundos"""
        titulo = Text("METODOLOGIA: 6 ALGORITMOS IMPLEMENTADOS", 
                     font_size=24, color=BLUE)
        titulo.to_edge(UP)
        
        principios = Text("Principios: Reproducibilidad - GridSearchCV - Metricas clinicas", 
                         font_size=12, color=GRAY)
        principios.next_to(titulo, DOWN, buff=0.3)
        
        self.play(Write(titulo))
        self.play(Write(principios))
        self.wait(2)
        
        # MÉTODOS CONVEXOS - Lado izquierdo
        convexos_titulo = Text("METODOS CONVEXOS", font_size=20, color=GREEN)
        convexos_titulo.to_edge(LEFT, buff=0.5).shift(UP * 1.5)
        
        # Regresión Logística
        reg_log_titulo = Text("1. REGRESION LOGISTICA", font_size=16, color=WHITE)
        reg_log_desc1 = Text("Modela probabilidad con funcion logistica", font_size=11, color=GRAY)
        reg_log_desc2 = Text("P(y=1|x) = 1/(1 + e^(-θx))", font_size=11, color=YELLOW)
        reg_log_grupo = VGroup(reg_log_titulo, reg_log_desc1, reg_log_desc2)
        reg_log_grupo.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # SVM Lineal
        svm_lin_titulo = Text("2. SVM LINEAL", font_size=16, color=WHITE)
        svm_lin_desc1 = Text("Maximiza margen entre clases", font_size=11, color=GRAY)
        svm_lin_desc2 = Text("Encuentra hiperplano optimo", font_size=11, color=YELLOW)
        svm_lin_grupo = VGroup(svm_lin_titulo, svm_lin_desc1, svm_lin_desc2)
        svm_lin_grupo.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # Regresión Ridge
        ridge_titulo = Text("3. REGRESION RIDGE", font_size=16, color=WHITE)
        ridge_desc1 = Text("Anade regularizacion L2", font_size=11, color=GRAY)
        ridge_desc2 = Text("Controla overfitting", font_size=11, color=YELLOW)
        ridge_grupo = VGroup(ridge_titulo, ridge_desc1, ridge_desc2)
        ridge_grupo.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        metodos_convexos_completo = VGroup(reg_log_grupo, svm_lin_grupo, ridge_grupo)
        metodos_convexos_completo.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        metodos_convexos_completo.next_to(convexos_titulo, DOWN, buff=0.3)
        
        # MÉTODOS NO CONVEXOS - Lado derecho
        no_convexos_titulo = Text("METODOS NO CONVEXOS", font_size=20, color=RED)
        no_convexos_titulo.to_edge(RIGHT, buff=0.5).shift(UP * 1.5)
        
        # Redes Neuronales
        rn_titulo = Text("4. REDES NEURONALES", font_size=16, color=WHITE)
        rn_desc1 = Text("Aproxima funciones complejas", font_size=11, color=GRAY)
        rn_desc2 = Text("Arquitectura: 100→50→25", font_size=11, color=YELLOW)
        rn_grupo = VGroup(rn_titulo, rn_desc1, rn_desc2)
        rn_grupo.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # SVM RBF
        svm_rbf_titulo = Text("5. SVM RBF", font_size=16, color=WHITE)
        svm_rbf_desc1 = Text("Kernel radial (Gaussiano)", font_size=11, color=GRAY)
        svm_rbf_desc2 = Text("K(x,x') = exp(-γ||x-x'||²)", font_size=11, color=YELLOW)
        svm_rbf_grupo = VGroup(svm_rbf_titulo, svm_rbf_desc1, svm_rbf_desc2)
        svm_rbf_grupo.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # Algoritmos Genéticos
        ag_titulo = Text("6. ALGORITMOS GENETICOS", font_size=16, color=WHITE)
        ag_desc1 = Text("Emula evolucion natural", font_size=11, color=GRAY)
        ag_desc2 = Text("Poblacion=50, Generaciones=30", font_size=11, color=YELLOW)
        ag_grupo = VGroup(ag_titulo, ag_desc1, ag_desc2)
        ag_grupo.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        metodos_no_convexos_completo = VGroup(rn_grupo, svm_rbf_grupo, ag_grupo)
        metodos_no_convexos_completo.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        metodos_no_convexos_completo.next_to(no_convexos_titulo, DOWN, buff=0.3)
        
        # Animaciones detalladas
        self.play(Write(convexos_titulo))
        self.wait(1)
        
        for metodo in metodos_convexos_completo:
            self.play(Write(metodo))
            self.wait(1.5)
        
        self.play(Write(no_convexos_titulo))
        self.wait(1)
        
        for metodo in metodos_no_convexos_completo:
            self.play(Write(metodo))
            self.wait(1.5)
        
        self.wait(3)
    
    def preprocesamiento_detallado(self):
        """Preprocesamiento detallado - 45 segundos"""
        titulo = Text("PREPROCESAMIENTO DE DATOS", font_size=26, color=BLUE)
        titulo.to_edge(UP)
        
        # División estratificada
        division_titulo = Text("DIVISION ESTRATIFICADA", font_size=20, color=YELLOW)
        division_titulo.next_to(titulo, DOWN, buff=0.8)
        
        division_explicacion = VGroup(
            Text("80% Entrenamiento: 455 muestras", font_size=16, color=GREEN),
            Text("20% Prueba: 114 muestras", font_size=16, color=BLUE),
            Text("Estratificacion: Mantiene proporcion de clases", font_size=16, color=ORANGE),
            Text("Reduce sesgo en estimacion de rendimiento", font_size=14, color=GRAY)
        )
        division_explicacion.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        division_explicacion.next_to(division_titulo, DOWN, buff=0.5)
        
        # Normalización Z-Score
        normalizacion_titulo = Text("NORMALIZACION Z-SCORE", font_size=20, color=RED)
        normalizacion_titulo.next_to(division_explicacion, DOWN, buff=0.8)
        
        formula = Text("x_norm = (x - μ) / σ", font_size=18, color=YELLOW)
        formula.next_to(normalizacion_titulo, DOWN, buff=0.3)
        
        normalizacion_beneficios = VGroup(
            Text("Estandariza caracteristicas: media=0, desviacion=1", font_size=14, color=WHITE),
            Text("Evita dominancia de escalas grandes", font_size=14, color=GREEN),
            Text("Mejora convergencia de algoritmos", font_size=14, color=GREEN),
            Text("Requisito para SVM y Redes Neuronales", font_size=14, color=GREEN)
        )
        normalizacion_beneficios.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        normalizacion_beneficios.next_to(formula, DOWN, buff=0.4)
        
        # Animaciones
        self.play(Write(titulo))
        self.wait(1)
        
        self.play(Write(division_titulo))
        self.wait(1)
        for item in division_explicacion:
            self.play(Write(item))
            self.wait(0.8)
        
        self.play(Write(normalizacion_titulo))
        self.wait(1)
        self.play(Write(formula))
        self.wait(1)
        
        for beneficio in normalizacion_beneficios:
            self.play(Write(beneficio))
            self.wait(0.6)
        
        self.wait(3)
    
    def configuraciones_optimas(self):
        """Configuraciones óptimas - 50 segundos"""
        titulo = Text("CONFIGURACIONES OPTIMAS ENCONTRADAS", 
                     font_size=24, color=BLUE)
        titulo.to_edge(UP)
        
        gridsearch_info = Text("Optimizacion mediante GridSearchCV con validacion cruzada k-fold", 
                              font_size=14, color=GRAY)
        gridsearch_info.next_to(titulo, DOWN, buff=0.3)
        
        self.play(Write(titulo))
        self.play(Write(gridsearch_info))
        self.wait(2)
        
        # Tabla de configuraciones
        configuraciones_data = [
            ("Regresion Logistica", "C=0.1, solver='lbfgs'"),
            ("SVM Lineal", "C=0.1, kernel='linear'"),
            ("Regresion Ridge", "alpha=1.0"),
            ("Redes Neuronales", "layers=(100,50,25), alpha=0.0001"),
            ("SVM RBF", "C=10.0, gamma=0.01"),
            ("Algoritmos Geneticos", "pop=50, gen=30, mut=0.1")
        ]
        
        # Encabezados
        header_metodo = Text("METODO", font_size=16, color=YELLOW)
        header_config = Text("CONFIGURACION", font_size=16, color=YELLOW)
        
        header_metodo.move_to(LEFT * 3 + UP * 1.5)
        header_config.move_to(RIGHT * 1 + UP * 1.5)
        
        self.play(Write(header_metodo), Write(header_config))
        self.wait(1)
        
        # Filas de datos
        for i, (metodo, config) in enumerate(configuraciones_data):
            y_pos = 1 - (i * 0.4)
            
            metodo_text = Text(metodo, font_size=12, color=WHITE)
            config_text = Text(config, font_size=12, color=GREEN)
            
            metodo_text.move_to(LEFT * 3 + UP * y_pos)
            config_text.move_to(RIGHT * 1 + UP * y_pos)
            
            self.play(Write(metodo_text), Write(config_text))
            self.wait(0.6)
        
        # Análisis de patrones
        analisis_titulo = Text("ANALISIS DE PATRONES ENCONTRADOS", 
                              font_size=18, color=RED)
        analisis_titulo.to_edge(DOWN, buff=2)
        
        patrones = VGroup(
            Text("Metodos lineales: C=0.1 (regularizacion moderada)", font_size=12, color=WHITE),
            Text("SVM RBF: gamma=0.01 → comportamiento cuasi-lineal", font_size=12, color=YELLOW),
            Text("Confirma: Dataset es LINEALMENTE SEPARABLE", font_size=12, color=GREEN)
        )
        patrones.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        patrones.next_to(analisis_titulo, DOWN, buff=0.3)
        
        self.play(Write(analisis_titulo))
        self.wait(1)
        for patron in patrones:
            self.play(Write(patron))
            self.wait(1)
        
        self.wait(3)
    
    def resultados_experimentales(self):
        """Resultados experimentales completos - 70 segundos"""
        titulo = Text("RESULTADOS EXPERIMENTALES", font_size=26, color=BLUE)
        titulo.to_edge(UP)
        
        # Hallazgo principal
        hallazgo = Text("HALLAZGO PRINCIPAL: EMPATE TECNICO", 
                       font_size=20, color=YELLOW)
        empate_detalle = Text("SVM Lineal vs SVM RBF: 98.25% precision", 
                             font_size=16, color=GREEN)
        
        hallazgo.next_to(titulo, DOWN, buff=0.5)
        empate_detalle.next_to(hallazgo, DOWN, buff=0.3)
        
        self.play(Write(titulo))
        self.wait(1)
        self.play(Write(hallazgo))
        self.play(Write(empate_detalle))
        self.wait(2)
        
        # Crear gráfico de barras detallado
        metodos = ["SVM\nLineal", "SVM\nRBF", "Reg.\nLogistica", "Redes\nNeuronales", "Alg.\nGeneticos", "Reg.\nRidge"]
        precisiones = [98.25, 98.25, 97.37, 96.49, 96.49, 95.61]
        tiempos = [0.48, 3.45, 0.54, 13.24, 17.74, 0.09]
        colores = [GREEN, GREEN, YELLOW, RED, RED, ORANGE]
        
        # Ejes del gráfico
        ejes = VGroup()
        x_axis = Line(start=[-4.5, -1.5, 0], end=[4.5, -1.5, 0], color=WHITE)
        y_axis = Line(start=[-4.5, -1.5, 0], end=[-4.5, 1.5, 0], color=WHITE)
        ejes.add(x_axis, y_axis)
        
        # Etiquetas del eje Y
        y_labels = VGroup()
        for i, val in enumerate([95, 96, 97, 98, 99, 100]):
            label = Text(f"{val}%", font_size=10, color=WHITE)
            y_pos = -1.5 + (i * 0.5)
            label.next_to([-4.5, y_pos, 0], LEFT, buff=0.2)
            y_labels.add(label)
        
        self.play(Create(ejes))
        self.play(Write(y_labels))
        self.wait(1)
        
        # Crear barras con animación individual
        barras = VGroup()
        for i, (metodo, precision, tiempo, color) in enumerate(zip(metodos, precisiones, tiempos, colores)):
            x_pos = -4 + (i * 1.4)
            altura = (precision - 95) / 5 * 2.5
            
            barra = Rectangle(width=1, height=altura, color=color, fill_opacity=0.8)
            barra.next_to([x_pos, -1.5, 0], UP, buff=0, aligned_edge=DOWN)
            
            # Valor de precisión
            valor_precision = Text(f"{precision}%", font_size=9, color=WHITE)
            valor_precision.next_to(barra, UP, buff=0.1)
            
            # Tiempo de ejecución
            valor_tiempo = Text(f"{tiempo}s", font_size=8, color=GRAY)
            valor_tiempo.next_to(barra, DOWN, buff=0.1)
            
            # Etiqueta del método
            metodo_label = Text(metodo, font_size=8)
            metodo_label.next_to([x_pos, -1.5, 0], DOWN, buff=0.8)
            
            grupo_barra = VGroup(barra, valor_precision, valor_tiempo, metodo_label)
            barras.add(grupo_barra)
            
            # Animar cada barra
            self.play(Create(grupo_barra))
            self.wait(0.8)
        
        # Resaltar empate técnico
        self.play(
            barras[0][0].animate.set_color(YELLOW),
            barras[1][0].animate.set_color(YELLOW),
            run_time=2
        )
        
        # Conclusión del gráfico
        conclusion_grafico = Text("SVM Lineal: Misma precision, 7x mas rapido", 
                                 font_size=14, color=ORANGE)
        conclusion_grafico.to_edge(DOWN)
        self.play(Write(conclusion_grafico))
        
        self.wait(4)
    
    def analisis_matrices_completo(self):
        """Análisis completo de matrices de confusión - 60 segundos"""
        titulo = Text("ANALISIS CLINICO: MATRICES DE CONFUSION", 
                     font_size=24, color=BLUE)
        titulo.to_edge(UP)
        
        interpretacion = Text("Interpretacion clinica: Falsos positivos vs Falsos negativos", 
                             font_size=14, color=GRAY)
        interpretacion.next_to(titulo, DOWN, buff=0.3)
        
        self.play(Write(titulo))
        self.play(Write(interpretacion))
        self.wait(2)
        
        # SVM Lineal - Izquierda
        svm_lineal_titulo = Text("SVM LINEAL: BALANCE OPTIMO", 
                                font_size=18, color=GREEN)
        svm_lineal_titulo.to_edge(LEFT, buff=1).shift(UP * 1.2)
        
        # Matriz de confusión SVM Lineal
        matriz_lineal = self.crear_matriz_detallada([[70, 1], [1, 42]], GREEN)
        matriz_lineal.scale(0.9)
        matriz_lineal.next_to(svm_lineal_titulo, DOWN, buff=0.4)
        
        # Interpretación SVM Lineal
        interpretacion_lineal = VGroup(
            Text("1 Falso Positivo", font_size=12, color=RED),
            Text("1 Falso Negativo", font_size=12, color=RED),
            Text("Balance equilibrado", font_size=12, color=GREEN)
        )
        interpretacion_lineal.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        interpretacion_lineal.next_to(matriz_lineal, DOWN, buff=0.3)
        
        # SVM RBF - Derecha
        svm_rbf_titulo = Text("SVM RBF: SENSIBILIDAD PERFECTA", 
                             font_size=18, color=RED)
        svm_rbf_titulo.to_edge(RIGHT, buff=1).shift(UP * 1.2)
        
        # Matriz de confusión SVM RBF
        matriz_rbf = self.crear_matriz_detallada([[69, 2], [0, 43]], RED)
        matriz_rbf.scale(0.9)
        matriz_rbf.next_to(svm_rbf_titulo, DOWN, buff=0.4)
        
        # Interpretación SVM RBF
        interpretacion_rbf = VGroup(
            Text("2 Falsos Positivos", font_size=12, color=RED),
            Text("0 Falsos Negativos", font_size=12, color=GREEN),
            Text("Sensibilidad perfecta", font_size=12, color=GREEN)
        )
        interpretacion_rbf.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        interpretacion_rbf.next_to(matriz_rbf, DOWN, buff=0.3)
        
        # Animaciones
        self.play(Write(svm_lineal_titulo))
        self.play(Create(matriz_lineal))
        self.wait(1)
        for item in interpretacion_lineal:
            self.play(Write(item))
            self.wait(0.6)
        
        self.play(Write(svm_rbf_titulo))
        self.play(Create(matriz_rbf))
        self.wait(1)
        for item in interpretacion_rbf:
            self.play(Write(item))
            self.wait(0.6)
        
        # Métricas clínicas detalladas
        metricas_titulo = Text("METRICAS CLINICAS DERIVADAS", 
                              font_size=16, color=YELLOW)
        metricas_titulo.to_edge(DOWN, buff=2)
        
        # Tabla de métricas
        tabla_metricas = VGroup(
            # Encabezados
            VGroup(
                Text("METRICA", font_size=12, color=YELLOW),
                Text("SVM LINEAL", font_size=12, color=GREEN),
                Text("SVM RBF", font_size=12, color=RED)
            ).arrange(RIGHT, buff=2),
            
            # Sensibilidad
            VGroup(
                Text("Sensibilidad", font_size=11, color=WHITE),
                Text("97.67%", font_size=11, color=GREEN),
                Text("100.00%", font_size=11, color=RED)
            ).arrange(RIGHT, buff=2.3),
            
            # Especificidad
            VGroup(
                Text("Especificidad", font_size=11, color=WHITE),
                Text("98.59%", font_size=11, color=GREEN),
                Text("97.18%", font_size=11, color=RED)
            ).arrange(RIGHT, buff=2.2),
            
            # VPP
            VGroup(
                Text("VPP", font_size=11, color=WHITE),
                Text("97.67%", font_size=11, color=GREEN),
                Text("95.56%", font_size=11, color=RED)
            ).arrange(RIGHT, buff=2.8),
            
            # VPN
            VGroup(
                Text("VPN", font_size=11, color=WHITE),
                Text("98.59%", font_size=11, color=GREEN),
                Text("100.00%", font_size=11, color=RED)
            ).arrange(RIGHT, buff=2.8)
        )
        
        tabla_metricas.arrange(DOWN, buff=0.25)
        tabla_metricas.next_to(metricas_titulo, DOWN, buff=0.3)
        
        self.play(Write(metricas_titulo))
        self.wait(1)
        
        for fila in tabla_metricas:
            self.play(Write(fila))
            self.wait(0.8)
        
        # Interpretación clínica final
        interpretacion_final = Text("SVM Lineal: Mejor BALANCE clinico general", 
                                   font_size=14, color=ORANGE)
        interpretacion_final.to_edge(DOWN)
        self.play(Write(interpretacion_final))
        
        self.wait(4)
    
    def crear_matriz_detallada(self, datos, color):
        """Crear matriz de confusión detallada"""
        matriz = VGroup()
        
        # Título de matriz
        titulo_predicho = Text("Predicho", font_size=10, color=WHITE)
        titulo_predicho.shift(UP * 1.2)
        matriz.add(titulo_predicho)
        
        # Crear celdas con interpretación visual
        for i in range(2):
            for j in range(2):
                # Determinar color de fondo
                if i == j:  # Predicciones correctas
                    fill_color = color
                    fill_opacity = 0.4
                else:  # Errores
                    fill_color = RED
                    fill_opacity = 0.2
                
                celda = Square(side_length=0.7, color=color, 
                              fill_color=fill_color, fill_opacity=fill_opacity)
                valor = Text(str(datos[i][j]), font_size=18, color=WHITE)
                
                grupo_celda = VGroup(celda, valor)
                grupo_celda.move_to([j * 0.8 - 0.4, -i * 0.8 + 0.4, 0])
                matriz.add(grupo_celda)
        
        # Etiquetas
        etiquetas = ["Benigno", "Maligno"]
        
        # Etiquetas verticales (Actual)
        actual_label = Text("Actual", font_size=10, color=WHITE)
        actual_label.move_to([-1, 0, 0])
        actual_label.rotate(PI/2)
        matriz.add(actual_label)
        
        for i, etiqueta in enumerate(etiquetas):
            label_v = Text(etiqueta, font_size=9, color=WHITE)
            label_v.move_to([-0.7, 0.4 - i * 0.8, 0])
            matriz.add(label_v)
            
            label_h = Text(etiqueta, font_size=9, color=WHITE)
            label_h.move_to([i * 0.8 - 0.4, 0.9, 0])
            matriz.add(label_h)
        
        return matriz
    
    def curvas_roc_completas(self):
        """Curvas ROC completas - 60 segundos"""
        titulo = Text("ANALISIS DE CURVAS ROC: CAPACIDAD DISCRIMINATIVA", 
                     font_size=22, color=BLUE)
        titulo.to_edge(UP)
        
        explicacion = Text("AUC-ROC: Area bajo la curva = Probabilidad de clasificar correctamente", 
                          font_size=12, color=GRAY)
        explicacion.next_to(titulo, DOWN, buff=0.3)
        
        self.play(Write(titulo))
        self.play(Write(explicacion))
        self.wait(2)
        
        # Crear ejes ROC
        ejes = VGroup()
        x_axis = Line(start=[-3, -2, 0], end=[2, -2, 0], color=WHITE)
        y_axis = Line(start=[-3, -2, 0], end=[-3, 2, 0], color=WHITE)
        ejes.add(x_axis, y_axis)
        
        # Etiquetas de ejes
        x_label = Text("Tasa de Falsos Positivos (1-Especificidad)", 
                      font_size=11)
        x_label.next_to(ejes, DOWN, buff=0.3)
        
        y_label = Text("Tasa de Verdaderos Positivos (Sensibilidad)", 
                      font_size=11)
        y_label.next_to(ejes, LEFT, buff=0.3).rotate(PI/2)
        
        # Línea diagonal (clasificador aleatorio)
        diagonal = Line([-3, -2, 0], [2, 2, 0], color=GRAY, stroke_width=2)
        diagonal_label = Text("Clasificador aleatorio (AUC = 0.5)", 
                             font_size=9, color=GRAY)
        diagonal_label.next_to(diagonal, RIGHT, buff=0.1)
        
        self.play(Create(ejes))
        self.play(Write(x_label), Write(y_label))
        self.play(Create(diagonal))
        self.play(Write(diagonal_label))
        self.wait(2)
        
        # Curvas ROC de los métodos principales
        # Regresión Logística (mejor AUC = 0.9957)
        curva_reg_log = VMobject(color=BLUE, stroke_width=4)
        puntos_reg_log = [[-3, -2, 0], [-2.9, 0.8, 0], [-2.2, 1.5, 0], [-1.2, 1.8, 0], [2, 2, 0]]
        curva_reg_log.set_points_smoothly(puntos_reg_log)
        
        # SVM Lineal (AUC = 0.9937)
        curva_svm_lineal = VMobject(color=GREEN, stroke_width=4)
        puntos_lineal = [[-3, -2, 0], [-2.8, 0.6, 0], [-2.1, 1.4, 0], [-1.0, 1.7, 0], [2, 2, 0]]
        curva_svm_lineal.set_points_smoothly(puntos_lineal)
        
        # SVM RBF (AUC = 0.9897)
        curva_svm_rbf = VMobject(color=RED, stroke_width=4)
        puntos_rbf = [[-3, -2, 0], [-2.6, 0.4, 0], [-1.9, 1.2, 0], [-0.8, 1.6, 0], [2, 2, 0]]
        curva_svm_rbf.set_points_smoothly(puntos_rbf)
        
        # Leyenda detallada con ranking
        leyenda_titulo = Text("RESULTADOS AUC-ROC (RANKING)", 
                             font_size=14, color=YELLOW)
        leyenda_titulo.to_edge(RIGHT, buff=0.3).shift(UP * 1.8)
        
        leyenda_items = VGroup(
            Text("1° Reg. Logistica: 0.9957", font_size=11, color=BLUE),
            Text("2° SVM Lineal: 0.9937", font_size=11, color=GREEN),
            Text("3° SVM RBF: 0.9897", font_size=11, color=RED),
            Text("", font_size=6),  # Espaciado
            Text("Interpretacion AUC:", font_size=11, color=YELLOW),
            Text(">0.99 = Excelente", font_size=9, color=WHITE),
            Text("0.9-0.99 = Muy bueno", font_size=9, color=WHITE),
            Text("0.8-0.9 = Bueno", font_size=9, color=WHITE),
            Text("", font_size=6),  # Espaciado
            Text("TODOS: Excelente", font_size=11, color=GREEN),
            Text("discriminacion", font_size=11, color=GREEN)
        )
        leyenda_items.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        leyenda_items.next_to(leyenda_titulo, DOWN, buff=0.2)
        
        # Animaciones de curvas
        self.play(Write(leyenda_titulo))
        self.wait(1)
        
        self.play(Create(curva_reg_log))
        self.play(Write(leyenda_items[0]))
        self.wait(1.2)
        
        self.play(Create(curva_svm_lineal))
        self.play(Write(leyenda_items[1]))
        self.wait(1.2)
        
        self.play(Create(curva_svm_rbf))
        self.play(Write(leyenda_items[2]))
        self.wait(1.2)
        
        # Interpretación
        for item in leyenda_items[4:]:
            if item.text.strip():  # Skip empty items
                self.play(Write(item))
                self.wait(0.4)
        
        # Conclusión ROC
        conclusion_roc = Text("Diferencias minimas: TODOS los metodos son excelentes", 
                             font_size=12, color=GREEN)
        conclusion_roc.to_edge(DOWN)
        self.play(Write(conclusion_roc))
        
        self.wait(4)
    
    def discusion_profunda(self):
        """Discusión profunda - 60 segundos"""
        titulo = Text("DISCUSION: IMPLICACIONES DE LOS RESULTADOS", 
                     font_size=22, color=BLUE)
        titulo.to_edge(UP)
        
        self.play(Write(titulo))
        self.wait(1)
        
        # Hallazgo principal
        hallazgo_principal = Text("HALLAZGO PRINCIPAL: Dataset LINEALMENTE SEPARABLE", 
                                 font_size=18, color=YELLOW)
        hallazgo_principal.next_to(titulo, DOWN, buff=0.8)
        self.play(Write(hallazgo_principal))
        self.wait(2)
        
        # Evidencias del hallazgo
        evidencias_titulo = Text("EVIDENCIAS EXPERIMENTALES:", font_size=16, color=GREEN)
        evidencias_titulo.next_to(hallazgo_principal, DOWN, buff=0.8)
        
        evidencias = VGroup(
            Text("1. SVM RBF converge con gamma=0.01 (cuasi-lineal)", font_size=13, color=WHITE),
            Text("2. Metodos convexos igualan a no convexos", font_size=13, color=WHITE),
            Text("3. Redes Neuronales: arquitectura compleja innecesaria", font_size=13, color=WHITE),
            Text("4. Algoritmos Geneticos: exploracion exhaustiva inutil", font_size=13, color=WHITE)
        )
        evidencias.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        evidencias.next_to(evidencias_titulo, DOWN, buff=0.5)
        
        self.play(Write(evidencias_titulo))
        self.wait(1)
        
        for evidencia in evidencias:
            self.play(Write(evidencia))
            self.wait(1.2)
        
        # Implicaciones clínicas
        self.wait(1)
        self.clear()
        
        implicaciones_titulo = Text("IMPLICACIONES PARA IMPLEMENTACION CLINICA", 
                                   font_size=20, color=RED)
        implicaciones_titulo.to_edge(UP)
        
        implicaciones = VGroup(
            Text("IMPLEMENTACION: SVM Lineal optimo para clinicas", font_size=15, color=GREEN),
            Text("EFICIENCIA: 7x mas rapido que SVM RBF", font_size=15, color=YELLOW),
            Text("COSTO: Menor requerimiento computacional", font_size=15, color=BLUE),
            Text("INTERPRETABILIDAD: Coeficientes lineales explicables", font_size=15, color=ORANGE),
            Text("PRECISION: Sin sacrificio de exactitud diagnostica", font_size=15, color=PURPLE)
        )
        implicaciones.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        implicaciones.next_to(implicaciones_titulo, DOWN, buff=1)
        
        self.play(Write(implicaciones_titulo))
        self.wait(1)
        
        for implicacion in implicaciones:
            self.play(Write(implicacion))
            self.wait(1.4)
        
        # Validación del principio de parsimonia
        parsimonia_titulo = Text("PRINCIPIO DE PARSIMONIA VALIDADO", 
                                font_size=16, color=PURPLE)
        parsimonia_titulo.next_to(implicaciones, DOWN, buff=1)
        
        parsimonia_cita = Text("\"La explicacion mas simple que funciona es la correcta\"", 
                              font_size=12, color=WHITE)
        parsimonia_cita.next_to(parsimonia_titulo, DOWN, buff=0.3)
        
        aplicacion_parsimonia = Text("Aplicado: Metodos lineales > Metodos complejos", 
                                    font_size=14, color=GREEN)
        aplicacion_parsimonia.next_to(parsimonia_cita, DOWN, buff=0.3)
        
        self.play(Write(parsimonia_titulo))
        self.play(Write(parsimonia_cita))
        self.play(Write(aplicacion_parsimonia))
        
        self.wait(4)
    
    def conclusiones_finales(self):
        """Conclusiones finales extensas - 85 segundos"""
        titulo = Text("CONCLUSIONES Y RECOMENDACIONES FINALES", 
                     font_size=22, color=BLUE)
        titulo.to_edge(UP)
        
        self.play(Write(titulo))
        self.wait(1)
        
        # Conclusiones principales numeradas
        conclusiones_titulo = Text("CONCLUSIONES PRINCIPALES:", font_size=18, color=YELLOW)
        conclusiones_titulo.next_to(titulo, DOWN, buff=0.8)
        
        conclusiones_principales = VGroup(
            VGroup(
                Text("1. EMPATE TECNICO CONFIRMADO", font_size=15, color=GREEN),
                Text("   SVM Lineal vs SVM RBF: 98.25% precision", font_size=12, color=GRAY),
                Text("   Diferencia: SVM Lineal 7x mas eficiente", font_size=12, color=GRAY)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            
            VGroup(
                Text("2. SUPERIORIDAD DE METODOS CONVEXOS", font_size=15, color=YELLOW),
                Text("   Eficiencia superior sin perdida de precision", font_size=12, color=GRAY),
                Text("   Convergencia: 0.09-0.54s vs 3.45-17.74s", font_size=12, color=GRAY)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            
            VGroup(
                Text("3. DATASET LINEALMENTE SEPARABLE", font_size=15, color=BLUE),
                Text("   Validado experimental y teoricamente", font_size=12, color=GRAY),
                Text("   Metodos complejos innecesarios", font_size=12, color=GRAY)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            
            VGroup(
                Text("4. PRINCIPIO DE PARSIMONIA VALIDADO", font_size=15, color=ORANGE),
                Text("   Simplicidad algorítmica = mejores resultados", font_size=12, color=GRAY),
                Text("   Contradiccion a asunciones de complejidad", font_size=12, color=GRAY)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        )
        
        conclusiones_principales.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        conclusiones_principales.next_to(conclusiones_titulo, DOWN, buff=0.5)
        
        self.play(Write(conclusiones_titulo))
        self.wait(1)
        
        for conclusion in conclusiones_principales:
            self.play(Write(conclusion))
            self.wait(2)
        
        self.wait(2)
        self.clear()
        
        # Recomendaciones para la práctica
        recomendaciones_titulo = Text("RECOMENDACIONES PARA LA PRACTICA CLINICA", 
                                     font_size=20, color=RED)
        recomendaciones_titulo.to_edge(UP)
        
        recomendaciones = VGroup(
            Text("USAR SVM LINEAL como algoritmo de primera eleccion", font_size=16, color=GREEN),
            Text("PRIORIZAR eficiencia computacional en implementaciones", font_size=16, color=YELLOW),
            Text("VALIDAR separabilidad antes de metodos complejos", font_size=16, color=BLUE),
            Text("APLICAR misma metodologia a otros datasets medicos", font_size=16, color=ORANGE),
            Text("IMPLEMENTAR en sistemas de soporte para decision clinica", font_size=16, color=PURPLE)
        )
        recomendaciones.arrange(DOWN, aligned_edge=LEFT, buff=0.8)
        recomendaciones.next_to(recomendaciones_titulo, DOWN, buff=1)
        
        self.play(Write(recomendaciones_titulo))
        self.wait(1)
        
        for recomendacion in recomendaciones:
            self.play(Write(recomendacion))
            self.wait(1.5)
        
        # Impacto y contribución
        impacto_titulo = Text("CONTRIBUCION CIENTIFICA", font_size=18, color=PURPLE)
        impacto_titulo.next_to(recomendaciones, DOWN, buff=1)
        
        contribucion = VGroup(
            Text("Validacion experimental del principio de parsimonia", font_size=14, color=WHITE),
            Text("Demostracion de separabilidad lineal en cancer de mama", font_size=14, color=WHITE),
            Text("Guia para seleccion algoritmica en diagnostico medico", font_size=14, color=WHITE)
        )
        contribucion.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        contribucion.next_to(impacto_titulo, DOWN, buff=0.4)
        
        self.play(Write(impacto_titulo))
        self.wait(1)
        for item in contribucion:
            self.play(Write(item))
            self.wait(1)
        
        # Mensaje final impactante
        self.wait(2)
        
        mensaje_final_titulo = Text("MENSAJE FINAL", font_size=20, color=YELLOW)
        mensaje_final_titulo.to_edge(DOWN, buff=2.5)
        
        mensaje_final = VGroup(
            Text("Los metodos CONVEXOS proporcionan la combinacion ideal de:", 
                 font_size=14, color=WHITE),
            Text("PRECISION + EFICIENCIA + INTERPRETABILIDAD", 
                 font_size=18, color=YELLOW),
            Text("para salvar vidas en diagnostico de cancer de mama", 
                 font_size=14, color=WHITE)
        )
        mensaje_final.arrange(DOWN, buff=0.3)
        mensaje_final.next_to(mensaje_final_titulo, DOWN, buff=0.3)
        
        self.play(Write(mensaje_final_titulo))
        self.wait(1)
        for linea in mensaje_final:
            self.play(Write(linea))
            self.wait(1.5)
        
        # Créditos finales
        self.wait(3)
        creditos = VGroup(
            Text("Mario Wilfredo Ramirez Puma", font_size=14, color=YELLOW),
            Text("Universidad Nacional del Altiplano - Puno", font_size=10, color=GRAY),
            Text("Escuela Profesional de Ingenieria Estadistica e Informatica", 
                 font_size=8, color=GRAY),
            Text("2025", font_size=12, color=BLUE)
        )
        creditos.arrange(DOWN, buff=0.2)
        creditos.to_edge(DOWN, buff=0.5)
        
        for credito in creditos:
            self.play(Write(credito))
            self.wait(1)
        
        self.wait(5)