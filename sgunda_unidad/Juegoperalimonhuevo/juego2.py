import tkinter as tk
from tkinter import ttk, messagebox
import random
from collections import defaultdict, Counter

class JuegoCaramelos:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🍭 Simulador del Juego de Caramelos")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Tipos de caramelos
        self.tipos_caramelos = ['🍋 Limón', '🥚 Huevo', '🍐 Pera']
        
        # Variables del juego
        self.dificultad = tk.StringVar(value="basico")
        self.jugadores = []
        self.num_jugadores = 6
        self.grupos = []
        self.juego_iniciado = False
        
        self.crear_interfaz_principal()
        
    def crear_interfaz_principal(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        titulo = tk.Label(main_frame, text="🍭 JUEGO DE CARAMELOS 🍭", 
                         font=('Arial', 24, 'bold'), bg='#2c3e50', fg='#ecf0f1')
        titulo.pack(pady=(0, 20))
        
        # Frame de configuración
        self.config_frame = tk.LabelFrame(main_frame, text="⚙️ Configuración", 
                                         font=('Arial', 14, 'bold'), bg='#34495e', 
                                         fg='#ecf0f1', relief='groove', bd=3)
        self.config_frame.pack(fill='x', pady=(0, 20))
        
        self.crear_configuracion()
        
        # Frame del juego
        self.juego_frame = tk.Frame(main_frame, bg='#2c3e50')
        self.juego_frame.pack(fill='both', expand=True)
        
    def crear_configuracion(self):
        # Dificultad
        dif_frame = tk.Frame(self.config_frame, bg='#34495e')
        dif_frame.pack(pady=10)
        
        tk.Label(dif_frame, text="🎯 Selecciona la dificultad:", 
                font=('Arial', 12, 'bold'), bg='#34495e', fg='#ecf0f1').pack()
        
        opciones_frame = tk.Frame(dif_frame, bg='#34495e')
        opciones_frame.pack(pady=5)
        
        tk.Radiobutton(opciones_frame, text="🟢 Básico: 1 de cada tipo (Individual)", 
                      variable=self.dificultad, value="basico", 
                      font=('Arial', 11), bg='#34495e', fg='#ecf0f1',
                      selectcolor='#27ae60').pack(anchor='w', pady=2)
        
        tk.Radiobutton(opciones_frame, text="🔴 Avanzado: 2 de cada tipo (Grupos de 3)", 
                      variable=self.dificultad, value="avanzado", 
                      font=('Arial', 11), bg='#34495e', fg='#ecf0f1',
                      selectcolor='#e74c3c').pack(anchor='w', pady=2)
        
        # Número de jugadores
        jugadores_frame = tk.Frame(self.config_frame, bg='#34495e')
        jugadores_frame.pack(pady=10)
        
        tk.Label(jugadores_frame, text="👥 Número de jugadores:", 
                font=('Arial', 12, 'bold'), bg='#34495e', fg='#ecf0f1').pack(side='left')
        
        self.jugadores_var = tk.IntVar(value=6)
        jugadores_spin = tk.Spinbox(jugadores_frame, from_=3, to=12, 
                                   textvariable=self.jugadores_var, width=5, 
                                   font=('Arial', 12))
        jugadores_spin.pack(side='left', padx=10)
        
        # Botón iniciar
        tk.Button(self.config_frame, text="🚀 INICIAR JUEGO", 
                 command=self.iniciar_juego, font=('Arial', 14, 'bold'),
                 bg='#3498db', fg='white', relief='raised', bd=3,
                 padx=20, pady=10).pack(pady=15)
        
    def iniciar_juego(self):
        self.num_jugadores = self.jugadores_var.get()
        self.juego_iniciado = True
        
        # Limpiar frame del juego
        for widget in self.juego_frame.winfo_children():
            widget.destroy()
            
        # Crear jugadores
        self.crear_jugadores()
        
        # Crear interfaz del juego
        self.crear_interfaz_juego()
        
    def crear_jugadores(self):
        self.jugadores = []
        for i in range(self.num_jugadores):
            # Cada jugador recibe 3 caramelos aleatorios
            caramelos_iniciales = [random.choice(self.tipos_caramelos) for _ in range(3)]
            
            jugador = {
                'id': i,
                'nombre': f'Jugador {i+1}',
                'caramelos': caramelos_iniciales,
                'chupetines': 0,
                'comodines': 0,
                'grupo': None,
                'activo': True
            }
            self.jugadores.append(jugador)
            
        # Crear grupos si es modo avanzado
        if self.dificultad.get() == "avanzado":
            self.crear_grupos()
            
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
            
        # Jugadores restantes van al último grupo
        if jugadores_disponibles and self.grupos:
            for jugador_id in jugadores_disponibles:
                self.jugadores[jugador_id]['grupo'] = len(self.grupos) - 1
                self.grupos[-1].append(jugador_id)
                
    def crear_interfaz_juego(self):
        # Título del modo
        modo = "BÁSICO - Individual" if self.dificultad.get() == "basico" else "AVANZADO - Grupos"
        color = "#27ae60" if self.dificultad.get() == "basico" else "#e74c3c"
        
        titulo_modo = tk.Label(self.juego_frame, text=f"🎮 MODO {modo}", 
                              font=('Arial', 18, 'bold'), bg='#2c3e50', fg=color)
        titulo_modo.pack(pady=(0, 10))
        
        # Objetivo
        if self.dificultad.get() == "basico":
            objetivo = "🎯 Objetivo: Consigue 1 caramelo de cada tipo (🍋🥚🍐)"
        else:
            objetivo = "🎯 Objetivo: Cada jugador necesita 2 de cada tipo (🍋🍋🥚🥚🍐🍐)"
            
        tk.Label(self.juego_frame, text=objetivo, 
                font=('Arial', 12), bg='#2c3e50', fg='#ecf0f1').pack(pady=5)
        
        tk.Label(self.juego_frame, text="📦 Cada jugador recibió 3 caramelos al inicio", 
                font=('Arial', 10), bg='#2c3e50', fg='#95a5a6').pack(pady=2)
        
        # Botones de control
        self.crear_botones_control()
        
        # Área de jugadores
        self.crear_area_jugadores()
        
    def crear_botones_control(self):
        botones_frame = tk.Frame(self.juego_frame, bg='#2c3e50')
        botones_frame.pack(pady=15)
        
        # Botón intercambio aleatorio
        tk.Button(botones_frame, text="🎲 Intercambio Aleatorio", 
                 command=self.intercambio_aleatorio, 
                 font=('Arial', 11, 'bold'), bg='#9b59b6', fg='white',
                 padx=15, pady=8, relief='raised').pack(side='left', padx=5)
        
        # Botón intercambio inteligente
        tk.Button(botones_frame, text="🧠 Intercambio Inteligente", 
                 command=self.intercambio_inteligente, 
                 font=('Arial', 11, 'bold'), bg='#e67e22', fg='white',
                 padx=15, pady=8, relief='raised').pack(side='left', padx=5)
        
        # Botón verificar ganadores
        tk.Button(botones_frame, text="🏆 Verificar Ganadores", 
                 command=self.verificar_ganadores, 
                 font=('Arial', 11, 'bold'), bg='#f39c12', fg='white',
                 padx=15, pady=8, relief='raised').pack(side='left', padx=5)
        
        # Botón reiniciar
        tk.Button(botones_frame, text="🔄 Nuevo Juego", 
                 command=self.reiniciar_juego, 
                 font=('Arial', 11, 'bold'), bg='#34495e', fg='white',
                 padx=15, pady=8, relief='raised').pack(side='left', padx=5)
        
    def crear_area_jugadores(self):
        # Frame con scroll para jugadores
        canvas_frame = tk.Frame(self.juego_frame, bg='#2c3e50')
        canvas_frame.pack(fill='both', expand=True, pady=10)
        
        # Canvas y scrollbar
        canvas = tk.Canvas(canvas_frame, bg='#34495e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#34495e')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Crear contenido
        if self.dificultad.get() == "basico":
            self.mostrar_jugadores_basico()
        else:
            self.mostrar_grupos_avanzado()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Actualizar scroll después de un momento
        self.root.after(100, lambda: canvas.configure(scrollregion=canvas.bbox("all")))
        
    def mostrar_jugadores_basico(self):
        # Filtrar solo jugadores activos
        jugadores_activos = [j for j in self.jugadores if j.get('activo', True)]
        
        if not jugadores_activos:
            tk.Label(self.scrollable_frame, text="🎊 ¡Todos los jugadores han ganado!", 
                    font=('Arial', 16, 'bold'), bg='#34495e', fg='#27ae60').pack(pady=50)
            return
        
        # Grid de jugadores activos (3 columnas)
        cols = 3
        for i, jugador in enumerate(jugadores_activos):
            row = i // cols
            col = i % cols
            
            # Card del jugador
            card = tk.LabelFrame(self.scrollable_frame, text=f"👤 {jugador['nombre']}", 
                               font=('Arial', 11, 'bold'), bg='#ecf0f1', 
                               fg='#2c3e50', relief='raised', bd=2)
            card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew', ipadx=8, ipady=8)
            
            # Configurar grid
            self.scrollable_frame.columnconfigure(col, weight=1)
            
            # Caramelos
            caramelos_text = " ".join(jugador['caramelos']) if jugador['caramelos'] else "Sin caramelos"
            tk.Label(card, text=f"Caramelos:\n{caramelos_text}", 
                    font=('Arial', 10), bg='#ecf0f1', fg='#2c3e50',
                    justify='center').pack(pady=5)
            
            # Estado
            if self.verificar_conjunto_completo_basico(jugador['caramelos']):
                estado_text = "✅ ¡COMPLETO!"
                estado_color = "#27ae60"
            else:
                faltantes = self.obtener_faltantes_basico(jugador['caramelos'])
                estado_text = f"❌ Faltan: {' '.join(faltantes)}"
                estado_color = "#e74c3c"
                
            tk.Label(card, text=estado_text, 
                    font=('Arial', 9, 'bold'), fg=estado_color, bg='#ecf0f1').pack(pady=3)
            
            # Premios
            if jugador['chupetines'] > 0 or jugador['comodines'] > 0:
                premios_text = f"🍭 {jugador['chupetines']}  ⭐ {jugador['comodines']}"
                tk.Label(card, text=premios_text, 
                        font=('Arial', 10, 'bold'), bg='#ecf0f1', fg='#8e44ad').pack(pady=2)
        
        # Mostrar jugadores retirados al final
        jugadores_retirados = [j for j in self.jugadores if not j.get('activo', True)]
        if jugadores_retirados:
            # Separador
            tk.Label(self.scrollable_frame, text="🏆 JUGADORES GANADORES (Retirados)", 
                    font=('Arial', 12, 'bold'), bg='#34495e', fg='#f39c12').pack(pady=(20, 10))
            
            retirados_frame = tk.Frame(self.scrollable_frame, bg='#34495e')
            retirados_frame.pack(fill='x', padx=20)
            
            for i, jugador in enumerate(jugadores_retirados):
                retired_card = tk.Frame(retirados_frame, bg='#95a5a6', relief='raised', bd=1)
                retired_card.pack(side='left', padx=5, pady=5, fill='x', expand=True)
                
                tk.Label(retired_card, text=f"🏆 {jugador['nombre']}", 
                        font=('Arial', 10, 'bold'), bg='#95a5a6', fg='#2c3e50').pack(pady=3)
                tk.Label(retired_card, text="¡GANADOR!", 
                        font=('Arial', 8, 'bold'), bg='#95a5a6', fg='#27ae60').pack()
                tk.Label(retired_card, text=f"🍭 {jugador['chupetines']}", 
                        font=('Arial', 8), bg='#95a5a6', fg='#8e44ad').pack()
                
    def mostrar_grupos_avanzado(self):
        # Verificar si quedan jugadores activos
        jugadores_activos_total = [j for j in self.jugadores if j.get('activo', True)]
        
        if not jugadores_activos_total:
            tk.Label(self.scrollable_frame, text="🎊 ¡Todos los grupos han ganado!", 
                    font=('Arial', 16, 'bold'), bg='#34495e', fg='#27ae60').pack(pady=50)
            return
        
        for grupo_id, grupo_jugadores in enumerate(self.grupos):
            # Filtrar jugadores activos del grupo
            jugadores_activos_grupo = [j_id for j_id in grupo_jugadores if self.jugadores[j_id].get('activo', True)]
            
            if not jugadores_activos_grupo:
                continue  # Saltar grupos donde todos ganaron
            
            # Frame del grupo
            grupo_frame = tk.LabelFrame(self.scrollable_frame, 
                                      text=f"👥 GRUPO {grupo_id + 1}", 
                                      font=('Arial', 12, 'bold'), 
                                      bg='#ecf0f1', fg='#2c3e50',
                                      relief='raised', bd=3)
            grupo_frame.pack(fill='x', padx=15, pady=10)
            
            # Analizar recursos del grupo
            caramelos_totales_grupo = []
            for j_id in jugadores_activos_grupo:
                caramelos_totales_grupo.extend(self.jugadores[j_id]['caramelos'])
            
            contador_grupo = Counter(caramelos_totales_grupo)
            puede_ganar_alguien = all(contador_grupo[tipo] >= 2 for tipo in self.tipos_caramelos)
            
            # Estado del grupo
            if puede_ganar_alguien:
                estado_grupo = "🎯 ¡Puede ganar alguien del grupo!"
                color_grupo = "#f39c12"
            else:
                estado_grupo = "⏳ Necesita más caramelos como grupo..."
                color_grupo = "#e67e22"
            
            tk.Label(grupo_frame, text=estado_grupo, 
                    font=('Arial', 11, 'bold'), fg=color_grupo, bg='#ecf0f1').pack(pady=5)
            
            # Mostrar recursos totales del grupo
            recursos_text = f"📦 Recursos del grupo: "
            for tipo in self.tipos_caramelos:
                recursos_text += f"{tipo}×{contador_grupo[tipo]} "
            
            tk.Label(grupo_frame, text=recursos_text, 
                    font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d').pack(pady=2)
            
            # Jugadores activos del grupo en grid
            jugadores_frame = tk.Frame(grupo_frame, bg='#ecf0f1')
            jugadores_frame.pack(fill='x', padx=10, pady=5)
            
            for i, jugador_id in enumerate(jugadores_activos_grupo):
                jugador = self.jugadores[jugador_id]
                
                # Card individual del jugador
                jugador_card = tk.Frame(jugadores_frame, bg='white', relief='groove', bd=1)
                jugador_card.grid(row=0, column=i, padx=5, pady=5, sticky='nsew')
                
                jugadores_frame.columnconfigure(i, weight=1)
                
                # Nombre
                tk.Label(jugador_card, text=f"👤 {jugador['nombre']}", 
                        font=('Arial', 10, 'bold'), bg='white').pack(pady=3)
                
                # Caramelos individuales
                caramelos_text = " ".join(jugador['caramelos']) if jugador['caramelos'] else "Sin caramelos"
                tk.Label(jugador_card, text=f"{caramelos_text}", 
                        font=('Arial', 9), bg='white').pack(pady=2)
                
                # Estado individual vs recursos grupales
                contador_individual = Counter(jugador['caramelos'])
                faltantes_individuales = sum(max(0, 2 - contador_individual[tipo]) for tipo in self.tipos_caramelos)
                
                if faltantes_individuales == 0:
                    estado_text = "✅ Listo para ganar"
                    estado_color = "#27ae60"
                elif puede_ganar_alguien:
                    estado_text = f"🤝 Puede ganar con ayuda del grupo"
                    estado_color = "#f39c12"
                else:
                    estado_text = f"❌ Faltan: {faltantes_individuales} caramelos"
                    estado_color = "#e74c3c"
                    
                tk.Label(jugador_card, text=estado_text, 
                        font=('Arial', 8, 'bold'), fg=estado_color, bg='white').pack(pady=2)
                
                # Premios
                if jugador['chupetines'] > 0 or jugador['comodines'] > 0:
                    premios_text = f"🍭{jugador['chupetines']} ⭐{jugador['comodines']}"
                    tk.Label(jugador_card, text=premios_text, 
                            font=('Arial', 8), bg='white', fg='#8e44ad').pack()
            
            # Mostrar jugadores retirados del grupo si los hay
            jugadores_retirados_grupo = [j_id for j_id in grupo_jugadores if not self.jugadores[j_id].get('activo', True)]
            if jugadores_retirados_grupo:
                tk.Label(grupo_frame, text="🏆 Ganadores retirados de este grupo:", 
                        font=('Arial', 9, 'italic'), bg='#ecf0f1', fg='#7f8c8d').pack(pady=(5, 0))
                
                retirados_text = ", ".join([self.jugadores[j_id]['nombre'] for j_id in jugadores_retirados_grupo])
                tk.Label(grupo_frame, text=retirados_text, 
                        font=('Arial', 9, 'bold'), bg='#ecf0f1', fg='#27ae60').pack(pady=(0, 5))
                            
    def obtener_faltantes_basico(self, caramelos):
        tipos_actuales = set(caramelos)
        return [tipo for tipo in self.tipos_caramelos if tipo not in tipos_actuales]
    
    def verificar_conjunto_completo_basico(self, caramelos):
        tipos_necesarios = set(self.tipos_caramelos)
        tipos_actuales = set(caramelos)
        return tipos_necesarios.issubset(tipos_actuales)
    
    def verificar_conjunto_completo_avanzado(self, caramelos):
        contador = Counter(caramelos)
        return all(contador[tipo] >= 2 for tipo in self.tipos_caramelos)
    
    def intercambio_aleatorio(self):
        # Solo intercambios entre jugadores activos
        jugadores_activos = [j for j in self.jugadores if j.get('activo', True)]
        
        if len(jugadores_activos) < 2:
            messagebox.showinfo("Sin Jugadores", "No hay suficientes jugadores activos para intercambiar.")
            return
        
        intercambios = []
        for _ in range(random.randint(3, 6)):
            # Seleccionar dos jugadores activos aleatoriamente
            j1, j2 = random.sample(jugadores_activos, 2)
            
            if j1['caramelos'] and j2['caramelos']:
                caramelo1 = random.choice(j1['caramelos'])
                caramelo2 = random.choice(j2['caramelos'])
                
                j1['caramelos'].remove(caramelo1)
                j2['caramelos'].remove(caramelo2)
                
                j1['caramelos'].append(caramelo2)
                j2['caramelos'].append(caramelo1)
                
                intercambios.append(f"{j1['nombre']} ↔ {j2['nombre']}")
        
        self.actualizar_interfaz()
        messagebox.showinfo("🎲 Intercambios Aleatorios", 
                           f"Se realizaron {len(intercambios)} intercambios:\n\n" + 
                           "\n".join(intercambios))
    
    def intercambio_inteligente(self):
        intercambios = []
        jugadores_activos = [j for j in self.jugadores if j.get('activo', True)]
        
        # Usar comodines primero
        for ganador in jugadores_activos:
            if ganador['comodines'] > 0:
                while ganador['comodines'] > 0:
                    ayuda_realizada = False
                    for necesitado in jugadores_activos:
                        if necesitado['id'] != ganador['id'] and necesitado['chupetines'] == 0:
                            if self.dificultad.get() == "basico":
                                faltantes = self.obtener_faltantes_basico(necesitado['caramelos'])
                                if faltantes:
                                    tipo_a_dar = random.choice(faltantes)
                                    necesitado['caramelos'].append(tipo_a_dar)
                                    ganador['comodines'] -= 1
                                    intercambios.append(f"⭐ {ganador['nombre']} ayudó a {necesitado['nombre']} con {tipo_a_dar}")
                                    ayuda_realizada = True
                                    break
                            else:
                                contador = Counter(necesitado['caramelos'])
                                for tipo in self.tipos_caramelos:
                                    if contador[tipo] < 2:
                                        necesitado['caramelos'].append(tipo)
                                        ganador['comodines'] -= 1
                                        intercambios.append(f"⭐ {ganador['nombre']} ayudó a {necesitado['nombre']} con {tipo}")
                                        ayuda_realizada = True
                                        break
                                if ayuda_realizada:
                                    break
                    if not ayuda_realizada:
                        break
        
        # Intercambios normales beneficiosos
        for i, j1 in enumerate(jugadores_activos):
            for j2 in jugadores_activos[i+1:]:
                if j1['chupetines'] == 0 and j2['chupetines'] == 0:
                    if self.dificultad.get() == "basico":
                        intercambio = self.buscar_intercambio_basico(j1, j2)
                    else:
                        intercambio = self.buscar_intercambio_avanzado(j1, j2)
                        
                    if intercambio:
                        intercambios.append(intercambio)
        
        self.actualizar_interfaz()
        
        if intercambios:
            messagebox.showinfo("🧠 Intercambios Inteligentes", 
                               f"Intercambios realizados:\n\n" + "\n".join(intercambios))
        else:
            messagebox.showinfo("🧠 Intercambios Inteligentes", 
                               "No se encontraron intercambios beneficiosos.")
    
    def buscar_intercambio_basico(self, j1, j2):
        faltantes1 = self.obtener_faltantes_basico(j1['caramelos'])
        faltantes2 = self.obtener_faltantes_basico(j2['caramelos'])
        
        for tipo_necesario1 in faltantes1:
            if tipo_necesario1 in j2['caramelos']:
                for tipo_necesario2 in faltantes2:
                    if tipo_necesario2 in j1['caramelos']:
                        # Realizar intercambio
                        j1['caramelos'].remove(tipo_necesario2)
                        j2['caramelos'].remove(tipo_necesario1)
                        j1['caramelos'].append(tipo_necesario1)
                        j2['caramelos'].append(tipo_necesario2)
                        
                        return f"🔄 {j1['nombre']} ↔ {j2['nombre']}: {tipo_necesario2} ↔ {tipo_necesario1}"
        return None
    
    def buscar_intercambio_avanzado(self, j1, j2):
        contador1 = Counter(j1['caramelos'])
        contador2 = Counter(j2['caramelos'])
        
        for tipo1 in self.tipos_caramelos:
            for tipo2 in self.tipos_caramelos:
                if (tipo1 != tipo2 and 
                    contador1[tipo1] > 1 and contador1[tipo2] < 2 and
                    contador2[tipo2] > 1 and contador2[tipo1] < 2):
                    
                    # Realizar intercambio
                    j1['caramelos'].remove(tipo1)
                    j2['caramelos'].remove(tipo2)
                    j1['caramelos'].append(tipo2)
                    j2['caramelos'].append(tipo1)
                    
                    return f"🔄 {j1['nombre']} ↔ {j2['nombre']}: {tipo1} ↔ {tipo2}"
        return None
    
    def verificar_ganadores(self):
        nuevos_ganadores = []
        jugadores_a_retirar = []
        
        if self.dificultad.get() == "basico":
            for jugador in self.jugadores:
                if (self.verificar_conjunto_completo_basico(jugador['caramelos']) and 
                    jugador['chupetines'] == 0 and jugador.get('activo', True)):
                    
                    # Dar premio
                    jugador['chupetines'] = 1
                    jugador['comodines'] = 1
                    
                    # Remover caramelos usados
                    for tipo in self.tipos_caramelos:
                        if tipo in jugador['caramelos']:
                            jugador['caramelos'].remove(tipo)
                    
                    nuevos_ganadores.append(jugador['nombre'])
                    
                    # Usar comodines antes de retirarse
                    self.usar_comodines_antes_retiro(jugador)
                    
                    # Marcar para retiro
                    jugadores_a_retirar.append(jugador)
        else:
            # MODO AVANZADO: Los grupos pueden compartir recursos
            for grupo_id, grupo_jugadores in enumerate(self.grupos):
                # Solo considerar jugadores activos del grupo
                jugadores_activos = [j_id for j_id in grupo_jugadores if self.jugadores[j_id].get('activo', True)]
                
                if not jugadores_activos:
                    continue
                
                # Verificar si algún jugador del grupo puede ganar
                nuevo_ganador_grupo = self.verificar_ganador_en_grupo(jugadores_activos)
                
                if nuevo_ganador_grupo:
                    jugador_ganador = self.jugadores[nuevo_ganador_grupo]
                    
                    # Dar premio al ganador
                    jugador_ganador['chupetines'] = 1
                    jugador_ganador['comodines'] = 2
                    
                    nuevos_ganadores.append(jugador_ganador['nombre'])
                    
                    # Usar comodines para ayudar dentro del grupo primero
                    self.usar_comodines_en_grupo(jugador_ganador, jugadores_activos)
                    
                    # Si ya no tiene comodines o no puede ayudar más en el grupo, 
                    # usar comodines restantes para otros grupos
                    if jugador_ganador['comodines'] > 0:
                        self.usar_comodines_antes_retiro(jugador_ganador)
                    
                    # Marcar para retiro
                    jugadores_a_retirar.append(jugador_ganador)
        
        # Retirar jugadores ganadores
        for jugador in jugadores_a_retirar:
            jugador['activo'] = False
            jugador['caramelos'] = []  # Ya no tiene caramelos
            
        self.actualizar_interfaz()
        
        if nuevos_ganadores:
            comodines = 1 if self.dificultad.get() == "basico" else 2
            mensaje = f"🏆 ¡FELICIDADES!\n\nNuevos ganadores:\n"
            mensaje += "\n".join(f"• {nombre}" for nombre in nuevos_ganadores)
            mensaje += f"\n\nCada ganador recibió:\n🍭 1 Chupetín\n⭐ {comodines} Comodín(es)"
            
            if self.dificultad.get() == "avanzado":
                mensaje += "\n\n🤝 Los ganadores usaron sus comodines para ayudar a su grupo primero."
            else:
                mensaje += "\n\n🚪 Los ganadores se retiraron después de ayudar a otros."
                
            messagebox.showinfo("¡Tenemos Ganadores!", mensaje)
            
            # Verificar si todos ganaron
            self.verificar_fin_juego()
        else:
            messagebox.showinfo("Sin Nuevos Ganadores", 
                               "No hay nuevos ganadores en esta verificación.")
    
    def verificar_ganador_en_grupo(self, jugadores_activos_grupo):
        """Verifica si algún jugador del grupo puede ganar usando recursos compartidos"""
        # Recopilar todos los caramelos del grupo
        caramelos_totales_grupo = []
        for jugador_id in jugadores_activos_grupo:
            caramelos_totales_grupo.extend(self.jugadores[jugador_id]['caramelos'])
        
        contador_grupo = Counter(caramelos_totales_grupo)
        
        # Verificar si el grupo tiene suficientes recursos para que alguien gane
        if all(contador_grupo[tipo] >= 2 for tipo in self.tipos_caramelos):
            # Buscar al jugador que esté más cerca de ganar individualmente
            mejor_candidato = None
            menor_faltante = float('inf')
            
            for jugador_id in jugadores_activos_grupo:
                jugador = self.jugadores[jugador_id]
                if jugador['chupetines'] == 0:  # Solo considerar quien no ha ganado
                    contador_individual = Counter(jugador['caramelos'])
                    faltantes = sum(max(0, 2 - contador_individual[tipo]) for tipo in self.tipos_caramelos)
                    
                    if faltantes < menor_faltante:
                        menor_faltante = faltantes
                        mejor_candidato = jugador_id
            
            # Si encontramos un candidato, redistribuir caramelos del grupo para que gane
            if mejor_candidato is not None:
                self.redistribuir_caramelos_grupo(mejor_candidato, jugadores_activos_grupo)
                return mejor_candidato
        
        return None
    
    def redistribuir_caramelos_grupo(self, ganador_id, jugadores_grupo):
        """Redistribuye los caramelos del grupo para que el ganador tenga 2 de cada tipo"""
        ganador = self.jugadores[ganador_id]
        
        # Recopilar todos los caramelos del grupo (excepto del ganador)
        caramelos_disponibles = []
        otros_jugadores = []
        
        for jugador_id in jugadores_grupo:
            if jugador_id != ganador_id:
                otros_jugadores.append(self.jugadores[jugador_id])
                caramelos_disponibles.extend(self.jugadores[jugador_id]['caramelos'])
        
        # Limpiar caramelos de otros jugadores temporalmente
        for jugador in otros_jugadores:
            jugador['caramelos'] = []
        
        # Dar al ganador exactamente 2 de cada tipo
        contador_ganador = Counter(ganador['caramelos'])
        caramelos_usados = []
        
        for tipo in self.tipos_caramelos:
            necesita = max(0, 2 - contador_ganador[tipo])
            for _ in range(necesita):
                if tipo in caramelos_disponibles:
                    caramelos_disponibles.remove(tipo)
                    ganador['caramelos'].append(tipo)
                    caramelos_usados.append(tipo)
        
        # Distribuir caramelos restantes entre otros jugadores del grupo
        for i, caramelo in enumerate(caramelos_disponibles):
            jugador_destino = otros_jugadores[i % len(otros_jugadores)]
            jugador_destino['caramelos'].append(caramelo)
        
        # Mostrar información de la redistribución
        if caramelos_usados:
            messagebox.showinfo("🔄 Redistribución de Grupo", 
                               f"El grupo redistribuyó caramelos:\n" +
                               f"• {ganador['nombre']} recibió: {' '.join(caramelos_usados)}\n" +
                               f"• Ahora puede reclamar su chupetín")
    
    def usar_comodines_en_grupo(self, ganador, jugadores_activos_grupo):
        """El ganador usa comodines para ayudar primero a su grupo"""
        ayudas_grupo = []
        
        while ganador['comodines'] > 0:
            ayuda_realizada = False
            
            # Buscar compañeros de grupo que necesiten ayuda
            for jugador_id in jugadores_activos_grupo:
                if jugador_id != ganador['id']:
                    necesitado = self.jugadores[jugador_id]
                    if necesitado['chupetines'] == 0:  # Solo ayudar a quien no ha ganado
                        
                        # Verificar si el grupo tiene recursos para que este jugador gane
                        if self.puede_ganar_con_ayuda(jugador_id, jugadores_activos_grupo):
                            # Dar un caramelo que le ayude a completar su conjunto
                            contador = Counter(necesitado['caramelos'])
                            for tipo in self.tipos_caramelos:
                                if contador[tipo] < 2:
                                    necesitado['caramelos'].append(tipo)
                                    ganador['comodines'] -= 1
                                    ayudas_grupo.append(f"{ganador['nombre']} → {necesitado['nombre']}: {tipo}")
                                    ayuda_realizada = True
                                    break
                        if ayuda_realizada:
                            break
            
            if not ayuda_realizada:
                break  # No puede ayudar más a su grupo
        
        if ayudas_grupo:
            messagebox.showinfo("🤝 Ayuda Dentro del Grupo", 
                               f"{ganador['nombre']} ayudó a su grupo:\n\n" + 
                               "\n".join(ayudas_grupo))
    
    def puede_ganar_con_ayuda(self, jugador_id, jugadores_grupo):
        """Verifica si un jugador puede ganar con ayuda de su grupo"""
        # Contar caramelos totales del grupo
        caramelos_totales = []
        for j_id in jugadores_grupo:
            caramelos_totales.extend(self.jugadores[j_id]['caramelos'])
        
        contador_total = Counter(caramelos_totales)
        
        # Verificar si hay suficientes recursos
        return all(contador_total[tipo] >= 2 for tipo in self.tipos_caramelos)
    
    def usar_comodines_antes_retiro(self, ganador):
        """El ganador usa todos sus comodines para ayudar antes de retirarse"""
        ayudas_realizadas = []
        
        while ganador['comodines'] > 0:
            ayuda_realizada = False
            
            # Buscar jugadores activos que necesiten ayuda
            for necesitado in self.jugadores:
                if (necesitado['id'] != ganador['id'] and 
                    necesitado.get('activo', True) and
                    necesitado['chupetines'] == 0):  # Solo ayudar a quien no ha ganado
                    
                    if self.dificultad.get() == "basico":
                        faltantes = self.obtener_faltantes_basico(necesitado['caramelos'])
                        if faltantes:
                            tipo_a_dar = random.choice(faltantes)
                            necesitado['caramelos'].append(tipo_a_dar)
                            ganador['comodines'] -= 1
                            ayudas_realizadas.append(f"{ganador['nombre']} → {necesitado['nombre']}: {tipo_a_dar}")
                            ayuda_realizada = True
                            break
                    else:
                        contador = Counter(necesitado['caramelos'])
                        for tipo in self.tipos_caramelos:
                            if contador[tipo] < 2:
                                necesitado['caramelos'].append(tipo)
                                ganador['comodines'] -= 1
                                ayudas_realizadas.append(f"{ganador['nombre']} → {necesitado['nombre']}: {tipo}")
                                ayuda_realizada = True
                                break
                        if ayuda_realizada:
                            break
            
            if not ayuda_realizada:
                break  # No hay más jugadores que necesiten ayuda
        
        if ayudas_realizadas:
            messagebox.showinfo("💝 Ayuda Realizada", 
                               f"{ganador['nombre']} ayudó antes de retirarse:\n\n" + 
                               "\n".join(ayudas_realizadas))
    
    def verificar_fin_juego(self):
        """Verificar si todos los jugadores han ganado"""
        jugadores_activos = [j for j in self.jugadores if j.get('activo', True)]
        
        if not jugadores_activos:
            messagebox.showinfo("🎊 ¡JUEGO TERMINADO!", 
                               "¡Felicidades!\n\n🏆 Todos los jugadores ganaron\n" +
                               "🤝 El trabajo en equipo funcionó perfectamente\n\n" +
                               "¡Excelente demostración del juego de caramelos!")
            return True
        return False
                
    def actualizar_interfaz(self):
        # Limpiar área de jugadores y recrear
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        if self.dificultad.get() == "basico":
            self.mostrar_jugadores_basico()
        else:
            self.mostrar_grupos_avanzado()
            
        # Actualizar scroll
        self.root.after(100, lambda: self.scrollable_frame.master.configure(
            scrollregion=self.scrollable_frame.master.bbox("all")))
    
    def reiniciar_juego(self):
        # Limpiar todo y volver al inicio
        for widget in self.juego_frame.winfo_children():
            widget.destroy()
        self.juego_iniciado = False
        messagebox.showinfo("🔄 Juego Reiniciado", "¡Configura un nuevo juego!")
    
    def ejecutar(self):
        self.root.mainloop()

# Ejecutar el juego
if __name__ == "__main__":
    juego = JuegoCaramelos()
    juego.ejecutar()