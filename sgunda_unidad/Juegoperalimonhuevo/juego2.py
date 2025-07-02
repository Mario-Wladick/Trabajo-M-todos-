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
        return [random.choice(self.tipos_caramelos) for _ in range(3)]
    
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
        
        tk.Label(self.juego_frame, text="📦 Cada jugador recibió 3 caramelos aleatorios al inicio", 
                font=('Arial', 10), bg='#f0f8ff', fg='#7f8c8d').pack(pady=2)
        
        # Botones de acción arriba
        botones_frame = tk.Frame(self.juego_frame, bg='#f0f8ff')
        botones_frame.pack(pady=10)
        
        tk.Button(botones_frame, text="🎲 Intercambio Aleatorio", 
                 command=self.intercambio_aleatorio, 
                 font=('Arial', 11), bg='#9b59b6', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🧠 Intercambio Inteligente", 
                 command=self.intercambio_inteligente, 
                 font=('Arial', 11), bg='#e67e22', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🏆 Verificar Ganadores", 
                 command=self.verificar_ganadores_basico, 
                 font=('Arial', 11), bg='#f39c12', fg='white').pack(side='left', padx=5)
        
        # Frame para mostrar jugadores directamente (sin scroll complicado)
        jugadores_container = tk.Frame(self.juego_frame, bg='#f0f8ff')
        jugadores_container.pack(fill='both', expand=True, pady=10, padx=20)
        
        # Crear cards de jugadores directamente
        self.crear_cards_jugadores_basico(jugadores_container)
        
    def crear_cards_jugadores_basico(self, parent):
        # Organizar en grid más compacto - 3 columnas
        cols = 3
        for i, jugador in enumerate(self.jugadores):
            row = i // cols
            col = i % cols
            
            # Frame del jugador más compacto
            jugador_frame = tk.LabelFrame(parent, text=f"👤 {jugador['nombre']}", 
                                        font=('Arial', 10, 'bold'), bg='white', relief='groove', bd=2)
            jugador_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew', ipadx=5, ipady=5)
            
            # Configurar columnas para expandirse
            parent.columnconfigure(col, weight=1)
            parent.rowconfigure(row, weight=1)
            
            # Caramelos actuales
            caramelos_text = " ".join(jugador['caramelos'])
            tk.Label(jugador_frame, text=f"Caramelos: {caramelos_text}", 
                    font=('Arial', 9), bg='white', wraplength=180).pack(pady=3)
            
            # Estado del jugador
            estado = self.verificar_conjunto_completo_basico(jugador['caramelos'])
            color_estado = '#27ae60' if estado else '#e74c3c'
            texto_estado = '✅ ¡Conjunto completo!' if estado else '❌ Necesita más caramelos'
            
            tk.Label(jugador_frame, text=texto_estado, 
                    font=('Arial', 8, 'bold'), fg=color_estado, bg='white').pack(pady=2)
            
            # Chupetines y comodines
            premios_frame = tk.Frame(jugador_frame, bg='white')
            premios_frame.pack(fill='x', pady=2)
            
            if jugador['chupetines'] > 0:
                tk.Label(premios_frame, text=f"🍭 Chupetines: {jugador['chupetines']}", 
                        font=('Arial', 8), bg='white', fg='#8e44ad').pack(side='left')
            if jugador['comodines'] > 0:
                tk.Label(premios_frame, text=f"⭐ Comodines: {jugador['comodines']}", 
                        font=('Arial', 8), bg='white', fg='#f39c12').pack(side='right')
                
            # Mostrar estado especial si tiene comodines
            if jugador['comodines'] > 0:
                tk.Label(jugador_frame, text="💡 Puede ayudar a otros", 
                        font=('Arial', 7, 'italic'), bg='white', fg='#16a085').pack()
                
    def crear_interfaz_avanzado(self):
        # Título del modo
        tk.Label(self.juego_frame, text="MODO AVANZADO - Grupos", 
                font=('Arial', 16, 'bold'), bg='#f0f8ff', fg='#8e44ad').pack(pady=5)
        
        tk.Label(self.juego_frame, text="🎯 Objetivo: Cada jugador del grupo necesita 2 de cada tipo (🍋🍋🥚🥚🍐🍐)", 
                font=('Arial', 12), bg='#f0f8ff').pack(pady=5)
        
        tk.Label(self.juego_frame, text="📦 Cada jugador recibió 3 caramelos aleatorios al inicio", 
                font=('Arial', 10), bg='#f0f8ff', fg='#7f8c8d').pack(pady=2)
        
        # Botones de acción arriba
        botones_frame = tk.Frame(self.juego_frame, bg='#f0f8ff')
        botones_frame.pack(pady=10)
        
        tk.Button(botones_frame, text="🎲 Intercambio Aleatorio", 
                 command=self.intercambio_aleatorio, 
                 font=('Arial', 11), bg='#9b59b6', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🧠 Intercambio Inteligente", 
                 command=self.intercambio_inteligente, 
                 font=('Arial', 11), bg='#e67e22', fg='white').pack(side='left', padx=5)
        
        tk.Button(botones_frame, text="🏆 Verificar Ganadores", 
                 command=self.verificar_ganadores_avanzado, 
                 font=('Arial', 11), bg='#f39c12', fg='white').pack(side='left', padx=5)
        
        # Frame para mostrar grupos directamente (sin scroll complicado)
        grupos_container = tk.Frame(self.juego_frame, bg='#f0f8ff')
        grupos_container.pack(fill='both', expand=True, pady=10, padx=20)
        
        # Crear cards de grupos directamente
        self.crear_cards_grupos_avanzado(grupos_container)
        
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
    
    def intercambio_inteligente(self):
        intercambios_realizados = []
        
        if self.dificultad.get() == "basico":
            intercambios_realizados = self.intercambio_inteligente_basico()
        else:
            intercambios_realizados = self.intercambio_inteligente_avanzado()
        
        self.actualizar_interfaz()
        
        if intercambios_realizados:
            mensaje = "🧠 Intercambios Inteligentes Realizados:\n\n"
            for intercambio in intercambios_realizados:
                mensaje += f"• {intercambio}\n"
            messagebox.showinfo("¡Intercambios Estratégicos!", mensaje)
        else:
            messagebox.showinfo("Sin Intercambios", "No se encontraron intercambios beneficiosos en este momento.")
    
    def intercambio_inteligente_basico(self):
        intercambios = []
        
        # Primero, identificar ganadores con comodines
        ganadores_con_comodines = [j for j in self.jugadores if j['comodines'] > 0]
        
        # Ayudar a otros jugadores usando comodines
        for ganador in ganadores_con_comodines:
            if ganador['comodines'] > 0:
                # Buscar jugadores que necesiten ayuda
                for necesitado in self.jugadores:
                    if necesitado['id'] != ganador['id'] and not self.verificar_conjunto_completo_basico(necesitado['caramelos']):
                        # Analizar qué le falta al jugador necesitado
                        caramelos_necesitado = set(necesitado['caramelos'])
                        tipos_faltantes = [tipo for tipo in self.tipos_caramelos if tipo not in caramelos_necesitado]
                        
                        if tipos_faltantes:
                            # El ganador usa su comodín para dar el caramelo que falta
                            tipo_a_dar = random.choice(tipos_faltantes)
                            ganador['comodines'] -= 1
                            necesitado['caramelos'].append(tipo_a_dar)
                            
                            intercambios.append(f"{ganador['nombre']} usó su comodín ⭐ para dar {tipo_a_dar} a {necesitado['nombre']}")
                            break
        
        # Intercambios normales entre jugadores
        for i, jugador1 in enumerate(self.jugadores):
            for j, jugador2 in enumerate(self.jugadores[i+1:], i+1):
                # Solo intercambiar si ninguno ha ganado ya
                if (not self.verificar_conjunto_completo_basico(jugador1['caramelos']) and 
                    not self.verificar_conjunto_completo_basico(jugador2['caramelos'])):
                    
                    intercambio_realizado = self.buscar_intercambio_beneficioso_basico(jugador1, jugador2)
                    if intercambio_realizado:
                        intercambios.append(intercambio_realizado)
        
        return intercambios
    
    def intercambio_inteligente_avanzado(self):
        intercambios = []
        
        # Trabajar por grupos
        for grupo_jugadores in self.grupos:
            # Identificar ganadores con comodines en el grupo
            ganadores_grupo = [self.jugadores[j_id] for j_id in grupo_jugadores if self.jugadores[j_id]['comodines'] > 0]
            
            # Ayudar dentro del grupo con comodines
            for ganador in ganadores_grupo:
                while ganador['comodines'] > 0:
                    ayuda_realizada = False
                    for jugador_id in grupo_jugadores:
                        necesitado = self.jugadores[jugador_id]
                        if (necesitado['id'] != ganador['id'] and 
                            not self.verificar_conjunto_completo_avanzado(necesitado['caramelos'])):
                            
                            # Analizar qué le falta (necesita 2 de cada tipo)
                            contador = defaultdict(int)
                            for caramelo in necesitado['caramelos']:
                                contador[caramelo] += 1
                            
                            for tipo in self.tipos_caramelos:
                                if contador[tipo] < 2:
                                    # Dar el caramelo que necesita
                                    ganador['comodines'] -= 1
                                    necesitado['caramelos'].append(tipo)
                                    intercambios.append(f"{ganador['nombre']} usó su comodín ⭐ para dar {tipo} a {necesitado['nombre']} (Grupo {necesitado['grupo']+1})")
                                    ayuda_realizada = True
                                    break
                        if ayuda_realizada:
                            break
                    if not ayuda_realizada:
                        break
            
            # Intercambios normales dentro del grupo
            for i, jugador_id1 in enumerate(grupo_jugadores):
                for jugador_id2 in grupo_jugadores[i+1:]:
                    jugador1 = self.jugadores[jugador_id1]
                    jugador2 = self.jugadores[jugador_id2]
                    
                    if (not self.verificar_conjunto_completo_avanzado(jugador1['caramelos']) and 
                        not self.verificar_conjunto_completo_avanzado(jugador2['caramelos'])):
                        
                        intercambio_realizado = self.buscar_intercambio_beneficioso_avanzado(jugador1, jugador2)
                        if intercambio_realizado:
                            intercambios.append(intercambio_realizado)
        
        return intercambios
    
    def buscar_intercambio_beneficioso_basico(self, jugador1, jugador2):
        # Analizar qué tiene cada uno y qué necesita
        tipos1 = set(jugador1['caramelos'])
        tipos2 = set(jugador2['caramelos'])
        
        necesita1 = [tipo for tipo in self.tipos_caramelos if tipo not in tipos1]
        necesita2 = [tipo for tipo in self.tipos_caramelos if tipo not in tipos2]
        
        # Buscar intercambio beneficioso
        for necesita_j1 in necesita1:
            if necesita_j1 in jugador2['caramelos']:
                for necesita_j2 in necesita2:
                    if necesita_j2 in jugador1['caramelos']:
                        # Realizar intercambio
                        jugador1['caramelos'].remove(necesita_j2)
                        jugador2['caramelos'].remove(necesita_j1)
                        jugador1['caramelos'].append(necesita_j1)
                        jugador2['caramelos'].append(necesita_j2)
                        
                        return f"{jugador1['nombre']} intercambió {necesita_j2} por {necesita_j1} con {jugador2['nombre']}"
        
        return None
    
    def buscar_intercambio_beneficioso_avanzado(self, jugador1, jugador2):
        # Contar caramelos de cada tipo
        contador1 = defaultdict(int)
        contador2 = defaultdict(int)
        
        for caramelo in jugador1['caramelos']:
            contador1[caramelo] += 1
        for caramelo in jugador2['caramelos']:
            contador2[caramelo] += 1
        
        # Buscar intercambios que mejoren la situación de ambos
        for tipo1 in self.tipos_caramelos:
            for tipo2 in self.tipos_caramelos:
                if tipo1 != tipo2:
                    # Jugador1 tiene exceso de tipo1 y necesita tipo2
                    # Jugador2 tiene exceso de tipo2 y necesita tipo1
                    if (contador1[tipo1] > 1 and contador1[tipo2] < 2 and
                        contador2[tipo2] > 1 and contador2[tipo1] < 2):
                        
                        # Realizar intercambio
                        jugador1['caramelos'].remove(tipo1)
                        jugador2['caramelos'].remove(tipo2)
                        jugador1['caramelos'].append(tipo2)
                        jugador2['caramelos'].append(tipo1)
                        
                        return f"{jugador1['nombre']} intercambió {tipo1} por {tipo2} con {jugador2['nombre']} (Grupo {jugador1['grupo']+1})"
        
        return None
    
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