# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ Generar datos
# ------------------------------------------------------------

# Semilla para reproducibilidad
np.random.seed(42)

# Número de puntos de datos
n = 15  # más puntos para ver mejor la interpolación

# Generamos x equiespaciados en [-3, 3]
x = np.linspace(-1.0, 1.0, n)

# Función original sin ruido
f = lambda x: np.exp(-x**2)

# Ruido gaussiano
noise = 0.001 * np.random.randn(n)

# Valores y con ruido
y = f(x) + noise

# ------------------------------------------------------------
# 2️⃣ Construir la matriz de evaluación para interpolación polinómica
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2️⃣ Construir la matriz de evaluación para interpolación polinómica (versión didáctica)
# ------------------------------------------------------------

# Inicializamos una matriz n x n de ceros
V = np.zeros((n, n))
# y llenamos la matriz de evaluaciones 
# Cada fila i corresponde a un punto x[i]
# Cada columna j corresponde a una función x[i] elevado a la potencia j
for i in range(n):
    for j in range(n):
        V[i, j] = x[i]**j

# ------------------------------------------------------------
# 3️⃣ Resolver los coeficientes del polinomio
# ------------------------------------------------------------

# Solución exacta (inversa explícita)
# c = V^-1 * y
c = np.linalg.inv(V) @ y

# Alternativamente, también se puede usar:
# c = np.linalg.solve(V, y)

# ------------------------------------------------------------
# 4️⃣ Evaluar el polinomio en puntos finos para graficar
# ------------------------------------------------------------

x_fino = np.linspace(-1, 1, 200)
# Evaluamos usando np.polyval requiere invertir coeficientes si increasing=True
y_poly = sum(c[i] * x_fino**i for i in range(n))

# ------------------------------------------------------------
# 5️⃣ Graficar resultados
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(x_fino, f(x_fino), 'g--', label='Función original $e^{-x^2}$')
plt.plot(x_fino, y_poly, 'b-', label='Polinomio interpolante')
plt.scatter(x, y, color='red', label='Datos con ruido')
plt.title('Interpolación polinómica 1D')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

#Intento 2: USAR interpolantes exponenciales centrados en los puntos 

# ------------------------------------------------------------
# Caso 2: Interpolación usando funciones base gaussianas
# y = sum_j c_j * exp(-(z - x_j)^2)
# ------------------------------------------------------------

# Semilla para reproducibilidad
np.random.seed(42)

# Número de puntos de datos
n = 20

# Generamos x equiespaciados en [-3, 3]
x = np.linspace(-3, 3, n)

# Función original sin ruido
f = lambda x: np.exp(-x**2)

# Ruido gaussiano
noise = 0.01 * np.random.randn(n)

# Valores y con ruido
y = f(x) + noise

# ------------------------------------------------------------
# Construir la matriz de evaluación con funciones base gaussianas
# Cada entrada A[i,j] = exp(-(x[i] - x[j])**2)
# ------------------------------------------------------------
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[i, j] = np.exp(-(x[i] - x[j])**2)

# ------------------------------------------------------------
# Resolver los coeficientes c
# ------------------------------------------------------------
c = np.linalg.solve(A, y)

# ------------------------------------------------------------
# Evaluar el interpolante en puntos finos
# ------------------------------------------------------------
z = np.linspace(-3, 3, 400)
y_interp = np.zeros_like(z)

for j in range(n):
    y_interp += c[j] * np.exp(-(z - x[j])**2)

# ------------------------------------------------------------
# Graficar resultados
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(z, f(z), 'g--', label='Función original $e^{-x^2}$')
plt.plot(z, y_interp, 'b-', label='Interpolante gaussiano')
plt.scatter(x, y, color='red', label='Datos con ruido')
plt.title('Interpolación 1D con funciones base gaussianas')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


