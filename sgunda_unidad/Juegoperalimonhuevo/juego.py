import tkinter as tk
from tkinter import ttk, messagebox
import random
from collections import defaultdict

class JuegoCaramelos:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simulador del Juego de Caramelos")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f8ff')
        
        # Tipos de caramelos
        self.tipos_caramelos = ['🍋 Limón', '🥚 Huevo', '🍐 Pera']
        
        # Variables del juego
        self.dificultad = tk.StringVar(value="basico")
        self.jugadores = []
        self.num_jugadores = 6
        self.grupos = []
        self.turno_actual = 0
        self.juego_iniciado = False
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Título principal
        titulo = tk.Label(self.root, text="🍭 JUEGO DE CARAMELOS 🍭", 
                         font=('Arial', 20, 'bold'), bg='#f0f8ff', fg='#2c3e50')
        titulo.pack(pady=10)
        
        # Frame de configuración
        config_frame = tk.LabelFrame(self.root, text="Configuración del Juego", 
                                   font=('Arial', 12, 'bold'), bg='#f0f8ff')
        config_frame.pack(pady=10, padx=20, fill='x')
        
        # Selección de dificultad
        tk.Label(config_frame, text="Selecciona la dificultad:", 
                font=('Arial', 11), bg='#f0f8ff').pack(pady=5)
        
        dif_frame = tk.Frame(config_frame, bg='#f0f8ff')
        dif_frame.pack(pady=5)
        
        tk.Radiobutton(dif_frame, text="Básico (Individual - 3 caramelos)", 
                      variable=self.dificultad, value="basico", 
                      font=('Arial', 10), bg='#f0f8ff').pack(anchor='w')
        tk.Radiobutton(dif_frame, text="Avanzado (Grupos de 3 - 6 caramelos)", 
                      variable=self.dificultad, value="avanzado", 
                      font=('Arial', 10), bg='#f0f8ff').pack(anchor='w')
        
        # Número de jugadores
        jugadores_frame = tk.Frame(config_frame, bg='#f0f8ff')
        jugadores_frame.pack(pady=5)
        
        tk.Label(jugadores_frame, text="Número de jugadores:", 
                font=('Arial', 11), bg='#f0f8ff').pack(side='left')
        self.jugadores_spin = tk.Spinbox(jugadores_frame, from_=3, to=12, 
                                       value=6, width=5, font=('Arial', 11))
        self.jugadores_spin.pack(side='left', padx=5)
        
        # Botón iniciar
        tk.Button(config_frame, text="🎮 INICIAR JUEGO", 
                 command=self.iniciar_juego, font=('Arial', 12, 'bold'),
                 bg='#27ae60', fg='white', relief='raised').pack(pady=10)
        
        # Frame principal del juego
        self.juego_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.juego_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
    def iniciar_juego(self):
        self.num_jugadores = int(self.jugadores_spin.get())
        self.juego_iniciado = True
        
        # Limpiar frame anterior
        for widget in self.juego_frame.winfo_children():
            widget.destroy()
            
        # Crear jugadores
        self.crear_jugadores()
        
        # Crear interfaz según dificultad
        if self.dificultad.get() == "basico":
            self.crear_interfaz_basico()
        else:
            self.crear_interfaz_avanzado()
            
    def crear_jugadores(self):
        self.jugadores = []
        for i in range(self.num_jugadores):
            jugador = {
                'id': i,
                'nombre': f'Jugador {i+1}',
                'caramelos': self.repartir_caramelos_iniciales(),
                'chupetines': 0,
                'comodines': 0,
                'grupo': None
            }
            self.jugadores.append(jugador)
            
        # Si es avanzado, crear grupos
        if self.dificultad.get() == "avanzado":
            self.crear_grupos()
            
    def repartir_caramelos_iniciales(self):
        return [random.choice(self.tipos_caramelos) for _ in range(2)]
    
    def crear_grupos(self):
        self.grupos = []
        jugadores_disponibles = list(range(self.num_jugadores))
        random.shuffle(jugadores_disponibles)
        
        grupo_id = 0
        while len(jugadores_disponibles) >= 3:
            grupo = jugadores_disponibles[:3]
            for jugador_id in grupo:
                self.jugadores[jugador_id]['grupo'] = grupo_id
            self.grupos.append(grupo)
            jugadores_disponibles = jugadores_disponibles[3:]
            grupo_id += 1
            
        # Jugadores restantes van al último grupo si es posible
        if jugadores_disponibles and self.grupos:
            for jugador_id in jugadores_disponibles:
                self.jugadores[jugador_id]['grupo'] = len(self.grupos) - 1
                self.grupos[-1].append(jugador_id)
                
    def crear_interfaz_basico(self):
        # Título del modo
        tk.Label(self.juego_frame, text="MODO BÁSICO - Individual", 
                font=('Arial', 16, 'bold'), bg='#f0f8ff', fg='#e74c3c').pack(pady=5)
        
        tk.Label(self.juego_frame, text="🎯 Objetivo: Consigue 1 caramelo de cada tipo (🍋🥚🍐)", 
                font=('Arial', 12), bg='#f0f8ff').pack(pady=5)
        
        # Botones de acción arriba
        botones_frame = tk.Frame(self.juego_frame, bg='#f0f8ff')
        botones_frame.pack(pady=10)
        
        tk.Button(botones_frame, text="🔄 Repartir Nuevos Caramelos", 
                 command=self.repartir_nuevos_caramelos, 
                 font=('Arial', 11), bg='#3498db', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🎲 Intercambio Aleatorio", 
                 command=self.intercambio_aleatorio, 
                 font=('Arial', 11), bg='#9b59b6', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🏆 Verificar Ganadores", 
                 command=self.verificar_ganadores_basico, 
                 font=('Arial', 11), bg='#f39c12', fg='white').pack(side='left', padx=5)
        
        # Frame principal con scroll
        main_container = tk.Frame(self.juego_frame, bg='#f0f8ff')
        main_container.pack(fill='both', expand=True, pady=10)
        
        # Canvas y scrollbar
        self.canvas_basico = tk.Canvas(main_container, bg='#f0f8ff', highlightthickness=0)
        scrollbar_v = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas_basico.yview)
        self.scrollable_frame_basico = tk.Frame(self.canvas_basico, bg='#f0f8ff')
        
        self.scrollable_frame_basico.bind(
            "<Configure>",
            lambda e: self.canvas_basico.configure(scrollregion=self.canvas_basico.bbox("all"))
        )
        
        self.canvas_basico.create_window((0, 0), window=self.scrollable_frame_basico, anchor="nw")
        self.canvas_basico.configure(yscrollcommand=scrollbar_v.set)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            self.canvas_basico.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas_basico.bind("<MouseWheel>", _on_mousewheel)
        
        # Crear cards de jugadores
        self.crear_cards_jugadores_basico(self.scrollable_frame_basico)
        
        # Pack canvas y scrollbar
        self.canvas_basico.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        
        # Actualizar scroll region después de crear contenido
        self.juego_frame.after(100, lambda: self.canvas_basico.configure(scrollregion=self.canvas_basico.bbox("all")))
        
    def crear_cards_jugadores_basico(self, parent):
        # Organizar en grid más compacto
        cols = 2
        for i, jugador in enumerate(self.jugadores):
            row = i // cols
            col = i % cols
            
            # Frame del jugador más compacto
            jugador_frame = tk.LabelFrame(parent, text=f"👤 {jugador['nombre']}", 
                                        font=('Arial', 10, 'bold'), bg='white', relief='groove', bd=2)
            jugador_frame.grid(row=row, column=col, padx=10, pady=8, sticky='ew', ipadx=5, ipady=5)
            
            # Configurar columnas para expandirse
            parent.columnconfigure(col, weight=1)
            
            # Caramelos actuales
            caramelos_text = " ".join(jugador['caramelos'])
            tk.Label(jugador_frame, text=f"Caramelos: {caramelos_text}", 
                    font=('Arial', 9), bg='white', wraplength=200).pack(pady=3)
            
            # Estado del jugador
            estado = self.verificar_conjunto_completo_basico(jugador['caramelos'])
            color_estado = '#27ae60' if estado else '#e74c3c'
            texto_estado = '✅ ¡Conjunto completo!' if estado else '❌ Necesita más caramelos'
            
            tk.Label(jugador_frame, text=texto_estado, 
                    font=('Arial', 8, 'bold'), fg=color_estado, bg='white').pack(pady=2)
            
            # Chupetines y comodines
            if jugador['chupetines'] > 0:
                tk.Label(jugador_frame, text=f"🍭 Chupetines: {jugador['chupetines']}", 
                        font=('Arial', 8), bg='white', fg='#8e44ad').pack()
            if jugador['comodines'] > 0:
                tk.Label(jugador_frame, text=f"⭐ Comodines: {jugador['comodines']}", 
                        font=('Arial', 8), bg='white', fg='#f39c12').pack()
        
        # Asegurar que el contenido se expanda
        for i in range(cols):
            parent.columnconfigure(i, weight=1)
                
    def crear_interfaz_avanzado(self):
        # Título del modo
        tk.Label(self.juego_frame, text="MODO AVANZADO - Grupos", 
                font=('Arial', 16, 'bold'), bg='#f0f8ff', fg='#8e44ad').pack(pady=5)
        
        tk.Label(self.juego_frame, text="🎯 Objetivo: Cada jugador del grupo necesita 2 de cada tipo (🍋🍋🥚🥚🍐🍐)", 
                font=('Arial', 12), bg='#f0f8ff').pack(pady=5)
        
        # Botones de acción arriba
        botones_frame = tk.Frame(self.juego_frame, bg='#f0f8ff')
        botones_frame.pack(pady=10)
        
        tk.Button(botones_frame, text="🔄 Repartir Nuevos Caramelos", 
                 command=self.repartir_nuevos_caramelos, 
                 font=('Arial', 11), bg='#3498db', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🎲 Intercambio Aleatorio", 
                 command=self.intercambio_aleatorio, 
                 font=('Arial', 11), bg='#9b59b6', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🏆 Verificar Ganadores", 
                 command=self.verificar_ganadores_avanzado, 
                 font=('Arial', 11), bg='#f39c12', fg='white').pack(side='left', padx=5)
        
        # Frame principal con scroll
        main_container = tk.Frame(self.juego_frame, bg='#f0f8ff')
        main_container.pack(fill='both', expand=True, pady=10)
        
        # Canvas y scrollbar
        self.canvas_avanzado = tk.Canvas(main_container, bg='#f0f8ff', highlightthickness=0)
        scrollbar_v = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas_avanzado.yview)
        self.scrollable_frame_avanzado = tk.Frame(self.canvas_avanzado, bg='#f0f8ff')
        
        self.scrollable_frame_avanzado.bind(
            "<Configure>",
            lambda e: self.canvas_avanzado.configure(scrollregion=self.canvas_avanzado.bbox("all"))
        )
        
        self.canvas_avanzado.create_window((0, 0), window=self.scrollable_frame_avanzado, anchor="nw")
        self.canvas_avanzado.configure(yscrollcommand=scrollbar_v.set)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            self.canvas_avanzado.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas_avanzado.bind("<MouseWheel>", _on_mousewheel)
        
        # Crear cards de grupos
        self.crear_cards_grupos_avanzado(self.scrollable_frame_avanzado)
        
        # Pack canvas y scrollbar
        self.canvas_avanzado.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        
        # Actualizar scroll region después de crear contenido
        self.juego_frame.after(100, lambda: self.canvas_avanzado.configure(scrollregion=self.canvas_avanzado.bbox("all")))
        
    def crear_cards_grupos_avanzado(self, parent):
        for grupo_id, grupo_jugadores in enumerate(self.grupos):
            # Frame del grupo
            grupo_frame = tk.LabelFrame(parent, text=f"👥 Grupo {grupo_id + 1}", 
                                      font=('Arial', 12, 'bold'), bg='#ecf0f1', relief='groove', bd=3)
            grupo_frame.pack(fill='x', padx=10, pady=10)
            
            # Frame interno para jugadores
            jugadores_frame = tk.Frame(grupo_frame, bg='#ecf0f1')
            jugadores_frame.pack(fill='x', padx=5, pady=5)
            
            # Estado del grupo
            grupo_completo = True
            for jugador_id in grupo_jugadores:
                jugador = self.jugadores[jugador_id]
                if not self.verificar_conjunto_completo_avanzado(jugador['caramelos']):
                    grupo_completo = False
                    break
                    
            color_grupo = '#27ae60' if grupo_completo else '#e74c3c'
            texto_grupo = '🏆 ¡Grupo Ganador!' if grupo_completo else '⏳ En progreso...'
            
            tk.Label(grupo_frame, text=texto_grupo, 
                    font=('Arial', 11, 'bold'), fg=color_grupo, bg='#ecf0f1').pack(pady=5)
            
            # Mostrar cada jugador del grupo
            for i, jugador_id in enumerate(grupo_jugadores):
                jugador = self.jugadores[jugador_id]
                
                jugador_frame = tk.Frame(jugadores_frame, bg='white', relief='raised', bd=1)
                jugador_frame.pack(fill='x', pady=2)
                
                # Información del jugador
                info_frame = tk.Frame(jugador_frame, bg='white')
                info_frame.pack(fill='x', padx=10, pady=5)
                
                tk.Label(info_frame, text=f"👤 {jugador['nombre']}", 
                        font=('Arial', 10, 'bold'), bg='white').pack(anchor='w')
                
                caramelos_text = " ".join(jugador['caramelos'])
                tk.Label(info_frame, text=f"Caramelos: {caramelos_text}", 
                        font=('Arial', 9), bg='white').pack(anchor='w')
                
                # Estado individual
                estado = self.verificar_conjunto_completo_avanzado(jugador['caramelos'])
                color_estado = '#27ae60' if estado else '#e74c3c'
                texto_estado = '✅ Completo' if estado else '❌ Incompleto'
                
                tk.Label(info_frame, text=texto_estado, 
                        font=('Arial', 9, 'bold'), fg=color_estado, bg='white').pack(anchor='w')
                
                # Premios
                if jugador['chupetines'] > 0 or jugador['comodines'] > 0:
                    premios_text = f"🍭 {jugador['chupetines']} ⭐ {jugador['comodines']}"
                    tk.Label(info_frame, text=premios_text, 
                            font=('Arial', 8), bg='white', fg='#8e44ad').pack(anchor='w')
                            
    def verificar_conjunto_completo_basico(self, caramelos):
        tipos_necesarios = set(self.tipos_caramelos)
        tipos_actuales = set(caramelos)
        return tipos_necesarios.issubset(tipos_actuales)
    
    def verificar_conjunto_completo_avanzado(self, caramelos):
        contador = defaultdict(int)
        for caramelo in caramelos:
            contador[caramelo] += 1
        
        for tipo in self.tipos_caramelos:
            if contador[tipo] < 2:
                return False
        return True
    
    def repartir_nuevos_caramelos(self):
        for jugador in self.jugadores:
            # Agregar 2 caramelos aleatorios
            nuevos_caramelos = [random.choice(self.tipos_caramelos) for _ in range(2)]
            jugador['caramelos'].extend(nuevos_caramelos)
            
        self.actualizar_interfaz()
        messagebox.showinfo("¡Caramelos Repartidos!", "Cada jugador recibió 2 nuevos caramelos aleatorios.")
    
    def intercambio_aleatorio(self):
        # Simular intercambios aleatorios entre jugadores
        for _ in range(random.randint(3, 8)):
            j1, j2 = random.sample(range(self.num_jugadores), 2)
            
            if (self.jugadores[j1]['caramelos'] and 
                self.jugadores[j2]['caramelos']):
                
                # Intercambiar un caramelo aleatorio
                caramelo1 = random.choice(self.jugadores[j1]['caramelos'])
                caramelo2 = random.choice(self.jugadores[j2]['caramelos'])
                
                self.jugadores[j1]['caramelos'].remove(caramelo1)
                self.jugadores[j2]['caramelos'].remove(caramelo2)
                
                self.jugadores[j1]['caramelos'].append(caramelo2)
                self.jugadores[j2]['caramelos'].append(caramelo1)
        
        self.actualizar_interfaz()
        messagebox.showinfo("¡Intercambios Realizados!", "Se realizaron varios intercambios aleatorios entre jugadores.")
    
    def verificar_ganadores_basico(self):
        ganadores = []
        for jugador in self.jugadores:
            if self.verificar_conjunto_completo_basico(jugador['caramelos']):
                if jugador['chupetines'] == 0:  # Solo si no ha ganado antes
                    jugador['chupetines'] += 1
                    jugador['comodines'] += 1
                    # Remover los caramelos usados
                    for tipo in self.tipos_caramelos:
                        if tipo in jugador['caramelos']:
                            jugador['caramelos'].remove(tipo)
                    ganadores.append(jugador['nombre'])
        
        self.actualizar_interfaz()
        
        if ganadores:
            mensaje = f"🏆 ¡Felicidades!\n\nGanadores de esta ronda:\n" + "\n".join(f"• {nombre}" for nombre in ganadores)
            mensaje += f"\n\nCada ganador recibió:\n• 🍭 1 Chupetín\n• ⭐ 1 Comodín"
            messagebox.showinfo("¡Tenemos Ganadores!", mensaje)
        else:
            messagebox.showinfo("Sin Ganadores", "Ningún jugador completó el conjunto esta ronda. ¡Sigan intercambiando!")
    
    def verificar_ganadores_avanzado(self):
        grupos_ganadores = []
        
        for grupo_id, grupo_jugadores in enumerate(self.grupos):
            todos_completos = True
            for jugador_id in grupo_jugadores:
                jugador = self.jugadores[jugador_id]
                if not self.verificar_conjunto_completo_avanzado(jugador['caramelos']):
                    todos_completos = False
                    break
            
            if todos_completos:
                # Verificar si ya ganaron antes
                ya_ganaron = any(self.jugadores[j_id]['chupetines'] > 0 for j_id in grupo_jugadores)
                
                if not ya_ganaron:
                    grupos_ganadores.append(grupo_id)
                    for jugador_id in grupo_jugadores:
                        jugador = self.jugadores[jugador_id]
                        jugador['chupetines'] += 1
                        jugador['comodines'] += 2
                        # Remover caramelos usados
                        caramelos_a_remover = []
                        for tipo in self.tipos_caramelos:
                            count = 0
                            for caramelo in jugador['caramelos']:
                                if caramelo == tipo and count < 2:
                                    caramelos_a_remover.append(caramelo)
                                    count += 1
                        for caramelo in caramelos_a_remover:
                            jugador['caramelos'].remove(caramelo)
        
        self.actualizar_interfaz()
        
        if grupos_ganadores:
            mensaje = f"🏆 ¡Felicidades!\n\nGrupos ganadores:\n"
            for grupo_id in grupos_ganadores:
                mensaje += f"• Grupo {grupo_id + 1}\n"
            mensaje += f"\nCada jugador ganador recibió:\n• 🍭 1 Chupetín\n• ⭐ 2 Comodines"
            messagebox.showinfo("¡Tenemos Grupos Ganadores!", mensaje)
        else:
            messagebox.showinfo("Sin Ganadores", "Ningún grupo completó todos los conjuntos. ¡Sigan intercambiando!")
    
    def actualizar_interfaz(self):
        # Limpiar contenido anterior
        for widget in self.juego_frame.winfo_children():
            widget.destroy()
            
        if self.dificultad.get() == "basico":
            self.crear_interfaz_basico()
        else:
            self.crear_interfaz_avanzado()
    
    def ejecutar(self):
        self.root.mainloop()

# Ejecutar el juego
if __name__ == "__main__":
    juego = JuegoCaramelos()
    juego.ejecutar()