# 📊 Proyecto IPISA - Regresión Lineal con Machine Learning

Aplicación desarrollada en Python para demostrar el funcionamiento de un modelo de **Regresión Lineal** utilizando Machine Learning.

El programa permite cargar datos manualmente mediante un archivo CSV, entrenar un modelo de regresión y realizar predicciones a partir de los datos introducidos.

---

## 🎯 Objetivo

El objetivo del proyecto es mostrar de una manera sencilla cómo funciona un modelo de Machine Learning capaz de encontrar relaciones entre diferentes variables y utilizar esas relaciones para realizar predicciones.

En este proyecto se utilizan datos relacionados con las ventas de un negocio.

El modelo utiliza:

- Inversión en capacitación
- Precio promedio
- Número de visitas

para estimar:

- Ventas

---

## 🧠 ¿Cómo funciona?

El programa sigue varios pasos:

1. 📂 Carga los datos desde un archivo CSV.
2. 🔍 Comprueba que los datos sean válidos.
3. 📊 Separa las variables de entrada de la variable que se desea predecir.
4. ✂️ Divide los datos en:
   - Datos de entrenamiento
   - Datos de prueba
5. 🤖 Entrena un modelo de Regresión Lineal.
6. 📈 Evalúa el rendimiento del modelo.
7. 🔮 Permite realizar nuevas predicciones.
8. 📊 Genera gráficos para visualizar los resultados.

---

## 📁 Estructura del proyecto

```text
Proyecto-ipisa/
│
├── Datos_proyecto.csv.py
├── LICENSE
├── Main.py
└── README.md