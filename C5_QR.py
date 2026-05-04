"""
Tutorial: Factorización QR

Descripción:
1) Factorización QR con NumPy
2) Implementación manual con reflexiones de Householder
3) Exploración de valores propios de matrices ortogonales

"""

import numpy as np
import matplotlib.pyplot as plt


# =========================
# EJERCICIO 1
# =========================
print("\n========================")
print("EJERCICIO 1: QR con NumPy")
print("========================")

#Definimos una matriz A
A = np.array([[1, 2],
                [3, 4],
                [5, 6]], dtype=float)

Q, R = np.linalg.qr(A)

print("\nMatriz A:")
print(A)

print("\nMatriz Q:")
print(Q)
print(Q.shape)

#verificamos que sea ortogonal:
Q2 = Q.transpose()
print(Q2@Q)
print(Q@Q2)
#Como se interpreta QQ^T?

print("\nMatriz R:")
print(R)

print("\nVerificación Q @ R:")
print(Q @ R)


# =========================
# EJERCICIO 2
# =========================

def householder_reflection(v):
    v = v.astype(float)
    n = len(v)

    e1 = np.zeros(n)
    e1[0] = 1.0

    alpha = np.linalg.norm(v)
    if v[0] >= 0:
        alpha = -alpha

    u = v - alpha * e1
    norm_u = np.linalg.norm(u)

    if norm_u == 0:
        return np.eye(n)

    u = u / norm_u
    H = np.eye(n) - 2 * np.outer(u, u)

    return H


def qr_householder(A):
    A = A.astype(float)
    m, n = A.shape
    R = A.copy()

    for k in range(n):
        x = R[k:, k]

        Hk = householder_reflection(x)

        H = np.eye(m)
        H[k:, k:] = Hk

        R = H @ R

    return R


print("\n========================")
print("EJERCICIO 2: Householder")
print("========================")

A = np.random.randn(5, 3)

R_manual = qr_householder(A)
_, R_numpy = np.linalg.qr(A, mode ="complete")
# Notar el mode="complete". Que hace si le quitamos ese parámetro? 

print("\nMatriz A:")
print(A)

print("\nR (manual):")
print(R_manual)

print("\nR (NumPy):")
print(R_numpy)

print("\nDiferencia absoluta:")
print(np.abs(R_manual - R_numpy))

#Ejercicio: Calcular los números de condición de A y de R


# =========================================================================
# EJERCICIO 3: 
# Como son los valores propios de matrices ortogonales aleatorias con medida de Haar?
# =========================================================================

def ejercicio_3(n=50):
    print("\n========================")
    print("EJERCICIO 3: Valores propios de matrices ortogonales aleatorias")
    print("========================")

    A = np.random.randn(n, n)
    #A es aleatoria con entradas independientes, gaussianas normales standard
    Q,R = np.linalg.qr(A, mode ="complete")
    #Q es una matriz ortogonal ALEATORIA, con la MEDIDA DE HAAR

    eigenvalues = np.linalg.eigvals(Q)

    print("\nPrimeros valores propios:")
    print(eigenvalues[:10])

    # Distribución en el plano complejo
    plt.figure(figsize=(6, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag, s=10)

    plt.axhline(0)
    plt.axvline(0)

    plt.title("Valores propios en el plano complejo")
    plt.xlabel("Parte real")
    plt.ylabel("Parte imaginaria")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



def uniform_points_in_unit_circle(n):
    # Sample angles uniformly
    theta = 2 * np.pi * np.random.rand(n)
    
    
    # Convert to Cartesian coordinates
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Plot points
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=10)
    
    # Draw unit circle boundary
    #circle = plt.Circle((0, 0), 1, fill=False)
    #plt.gca().add_patch(circle)
    
    plt.gca().set_aspect('equal')
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    plt.title(f"{n} Uniform Points in Unit Circle")
    plt.show()


#Ejecutar el ejercicio para diferentes valores de n
ejercicio_3(n=100)
uniform_points_in_unit_circle(n=100)
# Cómo se comparan los valores propios de matrices ortogonales 
# aleatorias con puntos uniformes en el círculo?
