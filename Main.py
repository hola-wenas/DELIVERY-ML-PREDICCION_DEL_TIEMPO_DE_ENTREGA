"""
============================================================
DELIVERY ML - PREDICCIÓN DEL TIEMPO DE ENTREGA
============================================================

Aplicación gráfica para entrenar un modelo de Regresión Lineal
que predice el tiempo de entrega de un pedido.

El usuario introduce sus propios datos manualmente.

VARIABLES DE ENTRADA:
    - Distancia del recorrido (km)
    - Peso del pedido (kg)
    - Cantidad de productos
    - Nivel de tráfico (1-5)
    - Hora del envío (0-23)

VARIABLE A PREDECIR:
    - Tiempo de entrega (minutos)

FUNCIONES:
    - Introducción manual de datos
    - Edición y eliminación de registros
    - Guardar datos en CSV
    - Cargar datos desde CSV
    - Entrenar modelo
    - Evaluar modelo
    - Realizar nuevas predicciones
    - Gráfico de valores reales vs predichos
============================================================
"""

# ============================================================
# 0. INSTALAR LIBRERÍAS SI SON NECESARIAS
# ============================================================

import subprocess
import sys


LIBRERIAS = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
]


for nombre_pip, nombre_import in LIBRERIAS:
    try:
        __import__(nombre_import)
    except ImportError:
        print(f"Instalando {nombre_pip}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", nombre_pip]
        )


# ============================================================
# 1. IMPORTACIONES
# ============================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

NOMBRE_PROGRAMA = "Delivery ML"
SUBTITULO = "Sistema de predicción del tiempo de entrega"

COLUMNAS_X = [
    "distancia_km",
    "peso_kg",
    "cantidad_productos",
    "nivel_trafico",
    "hora_envio",
]

COLUMNA_Y = "tiempo_entrega"


NOMBRES_DESCRIPTIVOS = {
    "distancia_km": "Distancia del recorrido (km)",
    "peso_kg": "Peso del pedido (kg)",
    "cantidad_productos": "Cantidad de productos",
    "nivel_trafico": "Nivel de tráfico (1-5)",
    "hora_envio": "Hora del envío (0-23)",
    "tiempo_entrega": "Tiempo de entrega (minutos)",
}


# ============================================================
# 3. COLORES
# ============================================================

COLOR_FONDO = "#F5F7FB"
COLOR_MENU = "#172033"
COLOR_MENU_HOVER = "#263653"

COLOR_PRIMARIO = "#2563EB"
COLOR_PRIMARIO_HOVER = "#1D4ED8"

COLOR_VERDE = "#16A34A"
COLOR_NARANJA = "#EA580C"
COLOR_ROJO = "#DC2626"

COLOR_TEXTO = "#172033"
COLOR_TEXTO_SUAVE = "#64748B"

COLOR_TARJETA = "#FFFFFF"
COLOR_BORDE = "#E2E8F0"


# ============================================================
# 4. APLICACIÓN PRINCIPAL
# ============================================================

class DeliveryMLApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Delivery ML - Predicción de tiempo de entrega"
        )

        self.root.geometry("1250x760")
        self.root.minsize(1050, 650)

        self.root.configure(bg=COLOR_FONDO)

        # ----------------------------------------------------
        # Variables internas
        # ----------------------------------------------------

        self.df = pd.DataFrame(
            columns=COLUMNAS_X + [COLUMNA_Y]
        )

        self.modelo = None

        self.mae = None
        self.rmse = None
        self.r2 = None

        self.predicciones = None
        self.valores_reales = None

        self.pagina_actual = "inicio"

        # ----------------------------------------------------
        # Estilos
        # ----------------------------------------------------

        self.configurar_estilos()

        # ----------------------------------------------------
        # Construir interfaz
        # ----------------------------------------------------

        self.crear_interfaz()

        self.mostrar_inicio()


    # ========================================================
    # ESTILOS
    # ========================================================

    def configurar_estilos(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            background="white",
            foreground=COLOR_TEXTO,
            rowheight=34,
            fieldbackground="white",
            borderwidth=0,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Treeview.Heading",
            background="#EEF2FF",
            foreground=COLOR_TEXTO,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
        )

        style.map(
            "Treeview",
            background=[
                ("selected", "#DBEAFE")
            ],
            foreground=[
                ("selected", COLOR_TEXTO)
            ],
        )

        style.configure(
            "TEntry",
            padding=8,
            font=("Segoe UI", 10),
        )

        style.configure(
            "TCombobox",
            padding=7,
            font=("Segoe UI", 10),
        )


    # ========================================================
    # INTERFAZ GENERAL
    # ========================================================

    def crear_interfaz(self):

        # ----------------------------------------------------
        # Menú lateral
        # ----------------------------------------------------

        self.menu = tk.Frame(
            self.root,
            bg=COLOR_MENU,
            width=230,
        )

        self.menu.pack(
            side="left",
            fill="y",
        )

        self.menu.pack_propagate(False)

        # Logo / título

        logo_frame = tk.Frame(
            self.menu,
            bg=COLOR_MENU,
        )

        logo_frame.pack(
            fill="x",
            padx=20,
            pady=(25, 30),
        )

        tk.Label(
            logo_frame,
            text="🚚",
            bg=COLOR_MENU,
            fg="white",
            font=("Segoe UI Emoji", 28),
        ).pack(
            anchor="w"
        )

        tk.Label(
            logo_frame,
            text="DELIVERY ML",
            bg=COLOR_MENU,
            fg="white",
            font=("Segoe UI", 17, "bold"),
        ).pack(
            anchor="w",
            pady=(4, 0),
        )

        tk.Label(
            logo_frame,
            text="Predicción inteligente",
            bg=COLOR_MENU,
            fg="#94A3B8",
            font=("Segoe UI", 9),
        ).pack(
            anchor="w"
        )

        # Botones

        self.botones_menu = {}

        self.crear_boton_menu(
            "📊  Inicio",
            "inicio",
        )

        self.crear_boton_menu(
            "📋  Datos",
            "datos",
        )

        self.crear_boton_menu(
            "🤖  Modelo",
            "modelo",
        )

        self.crear_boton_menu(
            "🔮  Predecir",
            "predecir",
        )

        # Información inferior

        separador = tk.Frame(
            self.menu,
            bg="#334155",
            height=1,
        )

        separador.pack(
            fill="x",
            padx=20,
            side="bottom",
            pady=(0, 60),
        )

        tk.Label(
            self.menu,
            text="Regresión Lineal\nMachine Learning",
            bg=COLOR_MENU,
            fg="#64748B",
            font=("Segoe UI", 9),
            justify="left",
        ).pack(
            side="bottom",
            anchor="w",
            padx=20,
            pady=(0, 20),
        )

        # ----------------------------------------------------
        # Área principal
        # ----------------------------------------------------

        self.contenido = tk.Frame(
            self.root,
            bg=COLOR_FONDO,
        )

        self.contenido.pack(
            side="right",
            fill="both",
            expand=True,
        )


    # ========================================================
    # BOTONES DEL MENÚ
    # ========================================================

    def crear_boton_menu(self, texto, pagina):

        boton = tk.Button(
            self.menu,
            text=texto,
            bg=COLOR_MENU,
            fg="#CBD5E1",
            activebackground=COLOR_MENU_HOVER,
            activeforeground="white",
            relief="flat",
            bd=0,
            anchor="w",
            padx=22,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=lambda: self.cambiar_pagina(pagina),
        )

        boton.pack(
            fill="x",
            ipady=11,
            pady=2,
        )

        self.botones_menu[pagina] = boton


    def cambiar_pagina(self, pagina):

        self.pagina_actual = pagina

        for nombre, boton in self.botones_menu.items():

            if nombre == pagina:

                boton.configure(
                    bg=COLOR_PRIMARIO,
                    fg="white",
                )

            else:

                boton.configure(
                    bg=COLOR_MENU,
                    fg="#CBD5E1",
                )

        if pagina == "inicio":
            self.mostrar_inicio()

        elif pagina == "datos":
            self.mostrar_datos()

        elif pagina == "modelo":
            self.mostrar_modelo()

        elif pagina == "predecir":
            self.mostrar_predecir()


    # ========================================================
    # LIMPIAR CONTENIDO
    # ========================================================

    def limpiar_contenido(self):

        for widget in self.contenido.winfo_children():
            widget.destroy()


    # ========================================================
    # CABECERA
    # ========================================================

    def crear_cabecera(self, titulo, descripcion):

        frame = tk.Frame(
            self.contenido,
            bg=COLOR_FONDO,
        )

        frame.pack(
            fill="x",
            padx=35,
            pady=(28, 20),
        )

        tk.Label(
            frame,
            text=titulo,
            bg=COLOR_FONDO,
            fg=COLOR_TEXTO,
            font=("Segoe UI", 23, "bold"),
        ).pack(
            anchor="w"
        )

        tk.Label(
            frame,
            text=descripcion,
            bg=COLOR_FONDO,
            fg=COLOR_TEXTO_SUAVE,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w",
            pady=(4, 0),
        )


    # ========================================================
    # TARJETA
    # ========================================================

    def crear_tarjeta(
        self,
        parent,
        titulo,
        valor,
        color=COLOR_PRIMARIO,
    ):

        tarjeta = tk.Frame(
            parent,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        tarjeta.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6,
        )

        tk.Label(
            tarjeta,
            text=titulo,
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO_SUAVE,
            font=("Segoe UI", 9, "bold"),
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 2),
        )

        etiqueta_valor = tk.Label(
            tarjeta,
            text=valor,
            bg=COLOR_TARJETA,
            fg=color,
            font=("Segoe UI", 22, "bold"),
        )

        etiqueta_valor.pack(
            anchor="w",
            padx=18,
            pady=(0, 16),
        )

        return etiqueta_valor


    # ========================================================
    # INICIO
    # ========================================================

    def mostrar_inicio(self):

        self.limpiar_contenido()

        self.crear_cabecera(
            "Panel de control",
            "Sistema de predicción del tiempo de entrega de pedidos.",
        )

        # Tarjetas

        tarjetas = tk.Frame(
            self.contenido,
            bg=COLOR_FONDO,
        )

        tarjetas.pack(
            fill="x",
            padx=30,
        )

        registros = len(self.df)

        self.crear_tarjeta(
            tarjetas,
            "REGISTROS DISPONIBLES",
            str(registros),
            COLOR_PRIMARIO,
        )

        self.crear_tarjeta(
            tarjetas,
            "ERROR PROMEDIO (MAE)",
            self.formatear_metrica(self.mae),
            COLOR_NARANJA,
        )

        self.crear_tarjeta(
            tarjetas,
            "ERROR RMSE",
            self.formatear_metrica(self.rmse),
            COLOR_ROJO,
        )

        self.crear_tarjeta(
            tarjetas,
            "R² DEL MODELO",
            self.formatear_r2(),
            COLOR_VERDE,
        )

        # ----------------------------------------------------
        # Información
        # ----------------------------------------------------

        panel = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        panel.pack(
            fill="both",
            expand=True,
            padx=36,
            pady=25,
        )

        tk.Label(
            panel,
            text="¿Cómo funciona el sistema?",
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO,
            font=("Segoe UI", 17, "bold"),
        ).pack(
            anchor="w",
            padx=25,
            pady=(25, 8),
        )

        texto = (
            "1. Introduce tus datos históricos de entregas.\n\n"
            "2. Entrena el modelo de regresión lineal.\n\n"
            "3. Revisa las métricas y el gráfico para evaluar el modelo.\n\n"
            "4. Introduce los datos de un nuevo pedido.\n\n"
            "5. El sistema estimará el tiempo de entrega en minutos."
        )

        tk.Label(
            panel,
            text=texto,
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO_SUAVE,
            justify="left",
            font=("Segoe UI", 11),
        ).pack(
            anchor="w",
            padx=25,
            pady=10,
        )


    # ========================================================
    # DATOS
    # ========================================================

    def mostrar_datos(self):

        self.limpiar_contenido()

        self.crear_cabecera(
            "Datos históricos",
            "Introduce los datos reales de tus pedidos para entrenar el modelo.",
        )

        # ----------------------------------------------------
        # Formulario
        # ----------------------------------------------------

        formulario = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        formulario.pack(
            fill="x",
            padx=36,
            pady=(0, 15),
        )

        campos = [
            ("Distancia del recorrido (km)", "distancia_km"),
            ("Peso del pedido (kg)", "peso_kg"),
            ("Cantidad de productos", "cantidad_productos"),
            ("Nivel de tráfico (1-5)", "nivel_trafico"),
            ("Hora del envío (0-23)", "hora_envio"),
            ("Tiempo de entrega (minutos)", "tiempo_entrega"),
        ]

        self.entradas_datos = {}

        for i, (texto, nombre) in enumerate(campos):

            columna = i % 3
            fila = i // 3

            bloque = tk.Frame(
                formulario,
                bg=COLOR_TARJETA,
            )

            bloque.grid(
                row=fila,
                column=columna,
                padx=15,
                pady=12,
                sticky="ew",
            )

            tk.Label(
                bloque,
                text=texto,
                bg=COLOR_TARJETA,
                fg=COLOR_TEXTO_SUAVE,
                font=("Segoe UI", 9, "bold"),
            ).pack(
                anchor="w"
            )

            entrada = ttk.Entry(
                bloque,
                width=28,
            )

            entrada.pack(
                fill="x",
                pady=(5, 0),
            )

            self.entradas_datos[nombre] = entrada

        for columna in range(3):
            formulario.columnconfigure(
                columna,
                weight=1,
            )

        # Botones

        botones = tk.Frame(
            formulario,
            bg=COLOR_TARJETA,
        )

        botones.grid(
            row=2,
            column=0,
            columnspan=3,
            pady=(5, 15),
        )

        self.boton(
            botones,
            "➕ Agregar registro",
            self.agregar_registro,
            COLOR_PRIMARIO,
        ).pack(
            side="left",
            padx=5,
        )

        self.boton(
            botones,
            "✏️ Editar seleccionado",
            self.editar_registro,
            "#475569",
        ).pack(
            side="left",
            padx=5,
        )

        self.boton(
            botones,
            "🗑️ Eliminar seleccionado",
            self.eliminar_registro,
            COLOR_ROJO,
        ).pack(
            side="left",
            padx=5,
        )

        # ----------------------------------------------------
        # Tabla
        # ----------------------------------------------------

        tabla_frame = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        tabla_frame.pack(
            fill="both",
            expand=True,
            padx=36,
            pady=(0, 15),
        )

        columnas_tabla = [
            "distancia_km",
            "peso_kg",
            "cantidad_productos",
            "nivel_trafico",
            "hora_envio",
            "tiempo_entrega",
        ]

        self.tabla = ttk.Treeview(
            tabla_frame,
            columns=columnas_tabla,
            show="headings",
        )

        encabezados = {
            "distancia_km": "Distancia (km)",
            "peso_kg": "Peso (kg)",
            "cantidad_productos": "Productos",
            "nivel_trafico": "Tráfico (1-5)",
            "hora_envio": "Hora envío",
            "tiempo_entrega": "Entrega (min)",
        }

        anchos = {
            "distancia_km": 130,
            "peso_kg": 110,
            "cantidad_productos": 130,
            "nivel_trafico": 120,
            "hora_envio": 110,
            "tiempo_entrega": 150,
        }

        for columna in columnas_tabla:

            self.tabla.heading(
                columna,
                text=encabezados[columna],
            )

            self.tabla.column(
                columna,
                width=anchos[columna],
                anchor="center",
            )

        scrollbar = ttk.Scrollbar(
            tabla_frame,
            orient="vertical",
            command=self.tabla.yview,
        )

        self.tabla.configure(
            yscrollcommand=scrollbar.set
        )

        self.tabla.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(15, 0),
            pady=15,
        )

        scrollbar.pack(
            side="right",
            fill="y",
            pady=15,
            padx=(0, 10),
        )

        self.tabla.bind(
            "<<TreeviewSelect>>",
            self.cargar_registro_seleccionado,
        )

        self.actualizar_tabla()

        # ----------------------------------------------------
        # Botones CSV
        # ----------------------------------------------------

        csv_frame = tk.Frame(
            self.contenido,
            bg=COLOR_FONDO,
        )

        csv_frame.pack(
            fill="x",
            padx=36,
            pady=(0, 15),
        )

        self.boton(
            csv_frame,
            "📂 Cargar CSV",
            self.cargar_csv,
            "#475569",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.boton(
            csv_frame,
            "💾 Guardar CSV",
            self.guardar_csv,
            COLOR_VERDE,
        ).pack(
            side="left",
            padx=8,
        )

        self.boton(
            csv_frame,
            "🧹 Limpiar todos",
            self.limpiar_datos,
            COLOR_ROJO,
        ).pack(
            side="right",
        )


    # ========================================================
    # BOTÓN GENERAL
    # ========================================================

    def boton(
        self,
        parent,
        texto,
        comando,
        color,
    ):

        return tk.Button(
            parent,
            text=texto,
            command=comando,
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
        )


    # ========================================================
    # OBTENER DATOS DEL FORMULARIO
    # ========================================================

    def obtener_datos_formulario(self):

        try:

            distancia = float(
                self.entradas_datos["distancia_km"].get()
            )

            peso = float(
                self.entradas_datos["peso_kg"].get()
            )

            productos = int(
                self.entradas_datos["cantidad_productos"].get()
            )

            trafico = int(
                self.entradas_datos["nivel_trafico"].get()
            )

            hora = float(
                self.entradas_datos["hora_envio"].get()
            )

            tiempo = float(
                self.entradas_datos["tiempo_entrega"].get()
            )

        except ValueError:

            messagebox.showerror(
                "Datos incorrectos",
                "Todos los campos deben contener valores numéricos.",
            )

            return None

        # ----------------------------------------------------
        # Validaciones
        # ----------------------------------------------------

        if distancia <= 0:

            messagebox.showerror(
                "Dato incorrecto",
                "La distancia debe ser mayor que 0.",
            )

            return None

        if peso < 0:

            messagebox.showerror(
                "Dato incorrecto",
                "El peso no puede ser negativo.",
            )

            return None

        if productos <= 0:

            messagebox.showerror(
                "Dato incorrecto",
                "La cantidad de productos debe ser mayor que 0.",
            )

            return None

        if trafico < 1 or trafico > 5:

            messagebox.showerror(
                "Dato incorrecto",
                "El nivel de tráfico debe estar entre 1 y 5.",
            )

            return None

        if hora < 0 or hora > 23:

            messagebox.showerror(
                "Dato incorrecto",
                "La hora debe estar entre 0 y 23.",
            )

            return None

        if tiempo <= 0:

            messagebox.showerror(
                "Dato incorrecto",
                "El tiempo de entrega debe ser mayor que 0.",
            )

            return None

        return {
            "distancia_km": distancia,
            "peso_kg": peso,
            "cantidad_productos": productos,
            "nivel_trafico": trafico,
            "hora_envio": hora,
            "tiempo_entrega": tiempo,
        }


    # ========================================================
    # AGREGAR REGISTRO
    # ========================================================

    def agregar_registro(self):

        datos = self.obtener_datos_formulario()

        if datos is None:
            return

        nueva_fila = pd.DataFrame([datos])

        self.df = pd.concat(
            [self.df, nueva_fila],
            ignore_index=True,
        )

        self.invalidar_modelo()

        self.actualizar_tabla()

        self.limpiar_formulario()

        messagebox.showinfo(
            "Registro agregado",
            "El pedido se agregó correctamente.",
        )


    # ========================================================
    # EDITAR REGISTRO
    # ========================================================

    def editar_registro(self):

        seleccion = self.tabla.selection()

        if not seleccion:

            messagebox.showwarning(
                "Selecciona un registro",
                "Selecciona primero el pedido que deseas editar.",
            )

            return

        datos = self.obtener_datos_formulario()

        if datos is None:
            return

        indice = int(
            self.tabla.item(
                seleccion[0],
                "text",
            )
            or self.tabla.index(seleccion[0])
        )

        # Como el índice visual coincide con el índice del DataFrame
        indice = self.tabla.index(seleccion[0])

        for columna in COLUMNAS_X + [COLUMNA_Y]:

            self.df.loc[
                indice,
                columna,
            ] = datos[columna]

        self.invalidar_modelo()

        self.actualizar_tabla()

        self.limpiar_formulario()

        messagebox.showinfo(
            "Registro actualizado",
            "El pedido se actualizó correctamente.",
        )


    # ========================================================
    # ELIMINAR REGISTRO
    # ========================================================

    def eliminar_registro(self):

        seleccion = self.tabla.selection()

        if not seleccion:

            messagebox.showwarning(
                "Selecciona un registro",
                "Selecciona primero el pedido que deseas eliminar.",
            )

            return

        confirmar = messagebox.askyesno(
            "Confirmar eliminación",
            "¿Quieres eliminar el registro seleccionado?",
        )

        if not confirmar:
            return

        indices = [
            self.tabla.index(item)
            for item in seleccion
        ]

        self.df = self.df.drop(
            indices
        ).reset_index(
            drop=True
        )

        self.invalidar_modelo()

        self.actualizar_tabla()

        self.limpiar_formulario()


    # ========================================================
    # LIMPIAR DATOS
    # ========================================================

    def limpiar_datos(self):

        if self.df.empty:
            return

        confirmar = messagebox.askyesno(
            "Eliminar todos los datos",
            "¿Estás seguro de que quieres eliminar todos los registros?",
        )

        if not confirmar:
            return

        self.df = pd.DataFrame(
            columns=COLUMNAS_X + [COLUMNA_Y]
        )

        self.invalidar_modelo()

        self.actualizar_tabla()

        self.limpiar_formulario()


    # ========================================================
    # FORMULARIO
    # ========================================================

    def limpiar_formulario(self):

        if not hasattr(self, "entradas_datos"):
            return

        for entrada in self.entradas_datos.values():

            entrada.delete(
                0,
                tk.END,
            )


    def cargar_registro_seleccionado(self, event=None):

        seleccion = self.tabla.selection()

        if not seleccion:
            return

        indice = self.tabla.index(
            seleccion[0]
        )

        fila = self.df.iloc[indice]

        for columna in COLUMNAS_X + [COLUMNA_Y]:

            entrada = self.entradas_datos[columna]

            entrada.delete(
                0,
                tk.END,
            )

            entrada.insert(
                0,
                str(fila[columna]),
            )


    # ========================================================
    # ACTUALIZAR TABLA
    # ========================================================

    def actualizar_tabla(self):

        if not hasattr(self, "tabla"):
            return

        for item in self.tabla.get_children():

            self.tabla.delete(item)

        for _, fila in self.df.iterrows():

            valores = [
                f"{fila['distancia_km']:.2f}",
                f"{fila['peso_kg']:.2f}",
                f"{int(fila['cantidad_productos'])}",
                f"{int(fila['nivel_trafico'])}",
                f"{fila['hora_envio']:.1f}",
                f"{fila['tiempo_entrega']:.2f}",
            ]

            self.tabla.insert(
                "",
                "end",
                values=valores,
            )


    # ========================================================
    # CSV
    # ========================================================

    def guardar_csv(self):

        if self.df.empty:

            messagebox.showwarning(
                "No hay datos",
                "No existen datos para guardar.",
            )

            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar datos",
            defaultextension=".csv",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Todos los archivos", "*.*"),
            ],
        )

        if not ruta:
            return

        try:

            self.df.to_csv(
                ruta,
                index=False,
            )

            messagebox.showinfo(
                "CSV guardado",
                f"Los datos se guardaron correctamente.\n\n{ruta}",
            )

        except Exception as error:

            messagebox.showerror(
                "Error",
                f"No se pudo guardar el archivo.\n\n{error}",
            )


    def cargar_csv(self):

        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Todos los archivos", "*.*"),
            ],
        )

        if not ruta:
            return

        try:

            datos = pd.read_csv(ruta)

            columnas_necesarias = (
                COLUMNAS_X + [COLUMNA_Y]
            )

            faltantes = [
                columna
                for columna in columnas_necesarias
                if columna not in datos.columns
            ]

            if faltantes:

                nombres = "\n".join(
                    NOMBRES_DESCRIPTIVOS[x]
                    for x in faltantes
                )

                messagebox.showerror(
                    "CSV incorrecto",
                    "Faltan estas columnas:\n\n"
                    + nombres,
                )

                return

            datos = datos[
                columnas_necesarias
            ].copy()

            for columna in columnas_necesarias:

                datos[columna] = pd.to_numeric(
                    datos[columna],
                    errors="coerce",
                )

            if datos.isnull().any().any():

                messagebox.showerror(
                    "Datos incorrectos",
                    "El CSV contiene valores vacíos o no numéricos.",
                )

                return

            self.df = datos.reset_index(
                drop=True
            )

            self.invalidar_modelo()

            self.actualizar_tabla()

            messagebox.showinfo(
                "CSV cargado",
                f"Se cargaron {len(self.df)} registros.",
            )

        except Exception as error:

            messagebox.showerror(
                "Error al cargar",
                f"No se pudo cargar el archivo.\n\n{error}",
            )


    # ========================================================
    # INVALIDAR MODELO
    # ========================================================

    def invalidar_modelo(self):

        self.modelo = None

        self.mae = None
        self.rmse = None
        self.r2 = None

        self.predicciones = None
        self.valores_reales = None


    # ========================================================
    # MODELO
    # ========================================================

    def mostrar_modelo(self):

        self.limpiar_contenido()

        self.crear_cabecera(
            "Modelo de Machine Learning",
            "Entrena y evalúa una regresión lineal utilizando tus datos históricos.",
        )

        # ----------------------------------------------------
        # Botón entrenar
        # ----------------------------------------------------

        panel_entrenar = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        panel_entrenar.pack(
            fill="x",
            padx=36,
            pady=(0, 20),
        )

        tk.Label(
            panel_entrenar,
            text="Entrenamiento del modelo",
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO,
            font=("Segoe UI", 15, "bold"),
        ).pack(
            anchor="w",
            padx=25,
            pady=(20, 5),
        )

        tk.Label(
            panel_entrenar,
            text=(
                "El sistema utiliza el 80% de los datos para entrenar "
                "y el 20% restante para evaluar el modelo."
            ),
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO_SUAVE,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w",
            padx=25,
        )

        self.boton(
            panel_entrenar,
            "🤖 Entrenar modelo",
            self.entrenar_modelo,
            COLOR_PRIMARIO,
        ).pack(
            anchor="w",
            padx=25,
            pady=20,
        )

        # ----------------------------------------------------
        # Métricas
        # ----------------------------------------------------

        tarjetas = tk.Frame(
            self.contenido,
            bg=COLOR_FONDO,
        )

        tarjetas.pack(
            fill="x",
            padx=30,
        )

        self.crear_tarjeta(
            tarjetas,
            "MAE - Error promedio",
            self.formatear_metrica(self.mae),
            COLOR_NARANJA,
        )

        self.crear_tarjeta(
            tarjetas,
            "RMSE - Error penalizado",
            self.formatear_metrica(self.rmse),
            COLOR_ROJO,
        )

        self.crear_tarjeta(
            tarjetas,
            "R² - Ajuste del modelo",
            self.formatear_r2(),
            COLOR_VERDE,
        )

        # ----------------------------------------------------
        # Coeficientes
        # ----------------------------------------------------

        panel_coef = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        panel_coef.pack(
            fill="both",
            expand=True,
            padx=36,
            pady=20,
        )

        tk.Label(
            panel_coef,
            text="Variables aprendidas por el modelo",
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO,
            font=("Segoe UI", 15, "bold"),
        ).pack(
            anchor="w",
            padx=25,
            pady=(20, 5),
        )

        self.texto_coeficientes = tk.Text(
            panel_coef,
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO_SUAVE,
            relief="flat",
            font=("Consolas", 10),
            height=10,
        )

        self.texto_coeficientes.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=15,
        )

        self.mostrar_coeficientes()


    # ========================================================
    # ENTRENAR
    # ========================================================

    def entrenar_modelo(self):

        if len(self.df) < 10:

            messagebox.showwarning(
                "Datos insuficientes",
                "Necesitas al menos 10 registros para entrenar el modelo.",
            )

            return

        try:

            X = self.df[COLUMNAS_X].astype(float)
            y = self.df[COLUMNA_Y].astype(float)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
            )

            self.modelo = LinearRegression()

            self.modelo.fit(
                X_train,
                y_train,
            )

            predicciones = self.modelo.predict(
                X_test
            )

            self.predicciones = predicciones
            self.valores_reales = y_test.to_numpy()

            self.mae = mean_absolute_error(
                y_test,
                predicciones,
            )

            self.rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predicciones,
                )
            )

            self.r2 = r2_score(
                y_test,
                predicciones,
            )

            self.mostrar_modelo()

            messagebox.showinfo(
                "Modelo entrenado",
                (
                    "El modelo se entrenó correctamente.\n\n"
                    f"Registros utilizados: {len(self.df)}\n"
                    f"Entrenamiento: {len(X_train)}\n"
                    f"Prueba: {len(X_test)}"
                ),
            )

        except Exception as error:

            messagebox.showerror(
                "Error de entrenamiento",
                f"No fue posible entrenar el modelo.\n\n{error}",
            )


    # ========================================================
    # COEFICIENTES
    # ========================================================

    def mostrar_coeficientes(self):

        if not hasattr(
            self,
            "texto_coeficientes",
        ):
            return

        self.texto_coeficientes.delete(
            "1.0",
            tk.END,
        )

        if self.modelo is None:

            self.texto_coeficientes.insert(
                tk.END,
                "Todavía no se ha entrenado ningún modelo.\n\n"
                "Ve a la sección Datos, introduce registros y "
                "después pulsa 'Entrenar modelo'.",
            )

            return

        texto = (
            "FÓRMULA APRENDIDA\n"
            "────────────────────────────────────────────\n\n"
        )

        texto += (
            f"Tiempo de entrega = "
            f"{self.modelo.intercept_:.3f}\n\n"
        )

        for nombre, coeficiente in zip(
            COLUMNAS_X,
            self.modelo.coef_,
        ):

            texto += (
                f"+ ({coeficiente:.3f}) × "
                f"{NOMBRES_DESCRIPTIVOS[nombre]}\n"
            )

        texto += (
            "\n────────────────────────────────────────────\n"
        )

        texto += (
            "\nEstos coeficientes fueron calculados a partir "
            "de los datos introducidos por el usuario."
        )

        self.texto_coeficientes.insert(
            tk.END,
            texto,
        )


    # ========================================================
    # PREDECIR
    # ========================================================

    def mostrar_predecir(self):

        self.limpiar_contenido()

        self.crear_cabecera(
            "Nueva predicción",
            "Introduce las características de un pedido para estimar su tiempo de entrega.",
        )

        # ----------------------------------------------------
        # Comprobar modelo
        # ----------------------------------------------------

        if self.modelo is None:

            panel = tk.Frame(
                self.contenido,
                bg=COLOR_TARJETA,
                highlightbackground=COLOR_BORDE,
                highlightthickness=1,
            )

            panel.pack(
                fill="x",
                padx=36,
                pady=20,
            )

            tk.Label(
                panel,
                text="⚠️ El modelo todavía no está entrenado",
                bg=COLOR_TARJETA,
                fg=COLOR_NARANJA,
                font=("Segoe UI", 16, "bold"),
            ).pack(
                pady=(25, 8),
            )

            tk.Label(
                panel,
                text=(
                    "Primero introduce suficientes datos y entrena "
                    "el modelo desde la sección 'Modelo'."
                ),
                bg=COLOR_TARJETA,
                fg=COLOR_TEXTO_SUAVE,
                font=("Segoe UI", 10),
            ).pack(
                pady=(0, 25),
            )

            return

        # ----------------------------------------------------
        # Formulario
        # ----------------------------------------------------

        panel = tk.Frame(
            self.contenido,
            bg=COLOR_TARJETA,
            highlightbackground=COLOR_BORDE,
            highlightthickness=1,
        )

        panel.pack(
            fill="x",
            padx=36,
            pady=10,
        )

        tk.Label(
            panel,
            text="Características del nuevo pedido",
            bg=COLOR_TARJETA,
            fg=COLOR_TEXTO,
            font=("Segoe UI", 16, "bold"),
        ).pack(
            anchor="w",
            padx=25,
            pady=(22, 15),
        )

        campos = [
            ("Distancia del recorrido (km)", "distancia_km"),
            ("Peso del pedido (kg)", "peso_kg"),
            ("Cantidad de productos", "cantidad_productos"),
            ("Nivel de tráfico (1-5)", "nivel_trafico"),
            ("Hora del envío (0-23)", "hora_envio"),
        ]

        self.entradas_prediccion = {}

        contenedor_campos = tk.Frame(
            panel,
            bg=COLOR_TARJETA,
        )

        contenedor_campos.pack(
            fill="x",
            padx=25,
        )

        for i, (texto, nombre) in enumerate(campos):

            bloque = tk.Frame(
                contenedor_campos,
                bg=COLOR_TARJETA,
            )

            bloque.grid(
                row=i // 2,
                column=i % 2,
                padx=10,
                pady=10,
                sticky="ew",
            )

            tk.Label(
                bloque,
                text=texto,
                bg=COLOR_TARJETA,
                fg=COLOR_TEXTO_SUAVE,
                font=("Segoe UI", 9, "bold"),
            ).pack(
                anchor="w"
            )

            entrada = ttk.Entry(
                bloque,
                width=40,
            )

            entrada.pack(
                fill="x",
                pady=(5, 0),
            )

            self.entradas_prediccion[nombre] = entrada

        contenedor_campos.columnconfigure(
            0,
            weight=1,
        )

        contenedor_campos.columnconfigure(
            1,
            weight=1,
        )

        self.boton(
            panel,
            "🔮 Calcular tiempo estimado",
            self.realizar_prediccion,
            COLOR_PRIMARIO,
        ).pack(
            pady=25,
        )

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        self.panel_resultado = tk.Frame(
            self.contenido,
            bg="#EFF6FF",
            highlightbackground="#BFDBFE",
            highlightthickness=1,
        )

        self.panel_resultado.pack(
            fill="x",
            padx=36,
            pady=20,
        )

        tk.Label(
            self.panel_resultado,
            text="TIEMPO ESTIMADO DE ENTREGA",
            bg="#EFF6FF",
            fg=COLOR_PRIMARIO,
            font=("Segoe UI", 11, "bold"),
        ).pack(
            pady=(22, 4),
        )

        self.etiqueta_resultado = tk.Label(
            self.panel_resultado,
            text="—",
            bg="#EFF6FF",
            fg=COLOR_TEXTO,
            font=("Segoe UI", 32, "bold"),
        )

        self.etiqueta_resultado.pack(
            pady=(0, 8),
        )

        tk.Label(
            self.panel_resultado,
            text=(
                "La estimación es calculada por el modelo "
                "utilizando los datos con los que fue entrenado."
            ),
            bg="#EFF6FF",
            fg=COLOR_TEXTO_SUAVE,
            font=("Segoe UI", 9),
        ).pack(
            pady=(0, 20),
        )


    # ========================================================
    # REALIZAR PREDICCIÓN
    # ========================================================

    def realizar_prediccion(self):

        try:

            datos = {}

            for nombre in COLUMNAS_X:

                valor = self.entradas_prediccion[
                    nombre
                ].get()

                datos[nombre] = float(valor)

        except ValueError:

            messagebox.showerror(
                "Datos incorrectos",
                "Todos los campos deben contener valores numéricos.",
            )

            return

        # ----------------------------------------------------
        # Validaciones
        # ----------------------------------------------------

        if datos["distancia_km"] <= 0:

            messagebox.showerror(
                "Dato incorrecto",
                "La distancia debe ser mayor que 0.",
            )

            return

        if datos["peso_kg"] < 0:

            messagebox.showerror(
                "Dato incorrecto",
                "El peso no puede ser negativo.",
            )

            return

        if datos["cantidad_productos"] <= 0:

            messagebox.showerror(
                "Dato incorrecto",
                "La cantidad de productos debe ser mayor que 0.",
            )

            return

        if not 1 <= datos["nivel_trafico"] <= 5:

            messagebox.showerror(
                "Dato incorrecto",
                "El tráfico debe estar entre 1 y 5.",
            )

            return

        if not 0 <= datos["hora_envio"] <= 23:

            messagebox.showerror(
                "Dato incorrecto",
                "La hora debe estar entre 0 y 23.",
            )

            return

        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------

        nuevo = pd.DataFrame(
            [datos],
            columns=COLUMNAS_X,
        )

        resultado = self.modelo.predict(
            nuevo
        )[0]

        if resultado < 0:
            resultado = 0

        self.etiqueta_resultado.configure(
            text=f"{resultado:.1f} minutos"
        )


    # ========================================================
    # GRÁFICO
    # ========================================================

    def mostrar_grafico(self):

        if (
            self.predicciones is None
            or self.valores_reales is None
        ):

            messagebox.showwarning(
                "Sin resultados",
                "Primero debes entrenar el modelo.",
            )

            return

        reales = np.asarray(
            self.valores_reales
        )

        predichos = np.asarray(
            self.predicciones
        )

        fondo = "#F8FAFC"
        azul = "#2563EB"
        naranja = "#EA580C"
        gris = "#64748B"

        fig, ax = plt.subplots(
            figsize=(10, 7),
            facecolor=fondo,
        )

        ax.set_facecolor(fondo)

        # ----------------------------------------------------
        # Límites
        # ----------------------------------------------------

        minimo = min(
            reales.min(),
            predichos.min(),
        )

        maximo = max(
            reales.max(),
            predichos.max(),
        )

        margen = (
            maximo - minimo
        ) * 0.08

        if margen == 0:
            margen = 1

        limites = (
            minimo - margen,
            maximo + margen,
        )

        ax.set_xlim(limites)
        ax.set_ylim(limites)

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        # ----------------------------------------------------
        # Línea perfecta
        # ----------------------------------------------------

        x_linea = np.array(
            limites
        )

        ax.plot(
            x_linea,
            x_linea,
            color=gris,
            linewidth=2,
            linestyle="--",
            label="Predicción perfecta",
        )

        # ----------------------------------------------------
        # Puntos
        # ----------------------------------------------------

        errores = np.abs(
            predichos - reales
        )

        dentro = (
            errores <= self.mae
        )

        ax.scatter(
            reales[dentro],
            predichos[dentro],
            s=80,
            color=azul,
            alpha=0.85,
            edgecolors="white",
            linewidth=1.2,
            label="Error dentro del MAE",
        )

        ax.scatter(
            reales[~dentro],
            predichos[~dentro],
            s=80,
            color=naranja,
            alpha=0.85,
            edgecolors="white",
            linewidth=1.2,
            label="Error fuera del MAE",
        )

        # ----------------------------------------------------
        # Títulos DESCRIPTIVOS
        # ----------------------------------------------------

        ax.set_title(
            "Tiempo de entrega real vs tiempo de entrega predicho",
            fontsize=17,
            fontweight="bold",
            color="#172033",
            pad=20,
        )

        ax.set_xlabel(
            "Tiempo de entrega real (minutos)",
            fontsize=11,
            color="#475569",
            labelpad=10,
        )

        ax.set_ylabel(
            "Tiempo de entrega predicho (minutos)",
            fontsize=11,
            color="#475569",
            labelpad=10,
        )

        # ----------------------------------------------------
        # Cuadrícula
        # ----------------------------------------------------

        ax.grid(
            True,
            color="#E2E8F0",
            linewidth=0.8,
        )

        ax.set_axisbelow(True)

        # ----------------------------------------------------
        # Métricas
        # ----------------------------------------------------

        texto_metricas = (
            f"R² = {self.r2:.3f}    |    "
            f"MAE = {self.mae:.2f} min    |    "
            f"RMSE = {self.rmse:.2f} min"
        )

        fig.text(
            0.5,
            0.02,
            texto_metricas,
            ha="center",
            fontsize=10,
            color="#64748B",
        )

        # ----------------------------------------------------
        # Leyenda
        # ----------------------------------------------------

        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
        )

        plt.tight_layout(
            rect=(0, 0.05, 1, 1)
        )

        # ----------------------------------------------------
        # Guardar
        # ----------------------------------------------------

        try:

            ruta = "grafico_tiempo_entrega.png"

            fig.savefig(
                ruta,
                dpi=200,
                bbox_inches="tight",
                facecolor=fondo,
            )

        except Exception:
            pass

        plt.show()


    # ========================================================
    # BOTÓN PARA GRÁFICO
    # ========================================================

    def mostrar_modelo_completo(self):

        self.mostrar_modelo()


    # ========================================================
    # FORMATEAR MÉTRICAS
    # ========================================================

    def formatear_metrica(self, valor):

        if valor is None:
            return "—"

        return f"{valor:.2f} min"


    def formatear_r2(self):

        if self.r2 is None:
            return "—"

        return f"{self.r2:.3f}"


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DeliveryMLApp(
        root
    )

    # --------------------------------------------------------
    # Atajo adicional:
    # doble clic en "Modelo" no es necesario.
    # Añadimos un botón de gráfico al menú contextual
    # mediante una tecla.
    # --------------------------------------------------------

    root.bind(
        "<Control-g>",
        lambda event: app.mostrar_grafico()
        if app.modelo is not None
        else None,
    )

    root.mainloop()