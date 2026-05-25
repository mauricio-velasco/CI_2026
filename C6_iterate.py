import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------
# 1. Construcción del sistema lineal
# ---------------------------------------------------

n = 20

# Matriz aleatoria
A = np.random.randn(n, n)

# Hacemos A diagonalmente dominante
# para garantizar convergencia
for i in range(n):
    A[i, i] = np.sum(np.abs(A[i])) + 1

# Vector b aleatorio
b = np.random.randn(n)

# Solución exacta (solo para medir error)
x_exacta = np.linalg.solve(A, b)

# Aproximación inicial
x0 = np.zeros(n)


# ---------------------------------------------------
# 2. Método de Jacobi
# ---------------------------------------------------

def jacobi(A, b, x0, max_iter=100, tol=1e-10):

    n = len(b)

    x = x0.copy()
    x_new = np.zeros_like(x)

    errores = []

    for k in range(max_iter):

        for i in range(n):

            suma = np.dot(A[i, :], x) - A[i, i] * x[i]

            x_new[i] = (b[i] - suma) / A[i, i]

        error = np.linalg.norm(x_new - x_exacta)

        errores.append(error)

        if error < tol:
            break

        x[:] = x_new

    return x_new, errores


# ---------------------------------------------------
# 3. Método de Gauss-Seidel
# ---------------------------------------------------

def gauss_seidel(A, b, x0, max_iter=100, tol=1e-10):

    n = len(b)

    x = x0.copy()

    errores = []

    for k in range(max_iter):

        x_old = x.copy()

        for i in range(n):

            suma1 = np.dot(A[i, :i], x[:i])

            suma2 = np.dot(A[i, i+1:], x_old[i+1:])

            x[i] = (b[i] - suma1 - suma2) / A[i, i]

        error = np.linalg.norm(x - x_exacta)

        errores.append(error)

        if error < tol:
            break

    return x, errores


# ---------------------------------------------------
# 4. Método SOR
# ---------------------------------------------------

def sor(A, b, x0, omega=1.2, max_iter=100, tol=1e-10):

    n = len(b)

    x = x0.copy()

    errores = []

    for k in range(max_iter):

        x_old = x.copy()

        for i in range(n):

            suma1 = np.dot(A[i, :i], x[:i])

            suma2 = np.dot(A[i, i+1:], x_old[i+1:])

            x[i] = ((1 - omega) * x_old[i]
                    + omega * (b[i] - suma1 - suma2) / A[i, i])

        error = np.linalg.norm(x - x_exacta)

        errores.append(error)

        if error < tol:
            break

    return x, errores


# ---------------------------------------------------
# 5. Ejecutar los métodos
# ---------------------------------------------------

x_jacobi, err_jacobi = jacobi(A, b, x0)

x_gs, err_gs = gauss_seidel(A, b, x0)

x_sor, err_sor = sor(A, b, x0, omega=1.2)


# ---------------------------------------------------
# 6. Graficar convergencia
# ---------------------------------------------------

plt.figure(figsize=(8,5))

plt.semilogy(err_jacobi, label='Jacobi')

plt.semilogy(err_gs, label='Gauss-Seidel')

plt.semilogy(err_sor, label='SOR ($\\omega=1.2$)')

plt.xlabel('Iteración')

plt.ylabel(r'Error $||x^{(k)} - x^*||$')

plt.title('Comparación de Métodos Iterativos')

plt.grid(True, which='both', linestyle='--')

plt.legend()

plt.show()

#Variantes:
#(1) Cambiar el punto inicial
#(2) Calcular el radio espectral de la matriz
#(3) Modificar la matriz para que no sea diagonal dominante y ver que sucede.


