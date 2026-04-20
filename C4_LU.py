import numpy as np
from scipy.linalg import lu, norm, lu_factor, lu_solve
import time

# ============================================================
# Experimento: costo de resolver Ax=b
# LU vs inversión explícita
# ============================================================

np.random.seed(0)

n = 800   # tamaño del sistema (ajustar según máquina)
k = 10    # número de vectores b (clave para ver la diferencia)

A = np.random.rand(n, n)
B = np.random.rand(n, k)   # múltiples RHS

print("\n===== Comparación de métodos para resolver Ax = B =====")

# ------------------------------------------------------------
# Método 1: Factorización LU + resolución
# ------------------------------------------------------------
start = time.time()

lu, piv = lu_factor(A)          # factorizar una sola vez
X_lu = lu_solve((lu, piv), B)   # resolver para todos los RHS

time_lu = time.time() - start

# ------------------------------------------------------------
# Método 2: Inversión explícita
# ------------------------------------------------------------
start = time.time()

A_inv = np.linalg.inv(A)
X_inv = A_inv @ B

time_inv = time.time() - start

# ------------------------------------------------------------
# Verificación de error
# ------------------------------------------------------------
error = np.linalg.norm(X_lu - X_inv)

print(f"Tamaño n = {n}, RHS = {k}")
print(f"Tiempo LU (factorizar + resolver): {time_lu:.4f} s")
print(f"Tiempo inversión explícita:       {time_inv:.4f} s")
print(f"Error ||X_lu - X_inv|| = {error:.2e}")

# ------------------------------------------------------------
# Experimento adicional: reutilizar LU
# ------------------------------------------------------------
print("\n--- Reutilizando la factorización LU ---")

B2 = np.random.rand(n, k)

start = time.time()
X_lu2 = lu_solve((lu, piv), B2)
time_lu_reuse = time.time() - start

start = time.time()
X_inv2 = A_inv @ B2
time_inv_reuse = time.time() - start

print(f"Tiempo LU (reutilizado): {time_lu_reuse:.4f} s")
print(f"Tiempo con A_inv:        {time_inv_reuse:.4f} s")

# ============================================================
# Experimento 2: Explorando sensibilidad de la factorización LU
# ============================================================

# Valores de epsilon a explorar
eps_values = [1, 1e-1, 1e-5, 1e-10,1e-15, 1e-20,1e-30]

print("===== Experimento con LU (mejores prácticas: SciPy/LAPACK) =====")

for eps in eps_values:
    # ------------------------------------------------------------
    # Definición de la matriz A(epsilon)
    # ------------------------------------------------------------
    A = np.array([[eps, 1.0],
                  [1.0, 1.0]])

    # ------------------------------------------------------------
    # Factorización LU con pivoteo parcial (implementación robusta)
    # SciPy devuelve P, L, U tal que: P A = L U
    # ------------------------------------------------------------
    P, L, U = lu(A)

    # ------------------------------------------------------------
    # Error de reconstrucción
    # ------------------------------------------------------------
    error = norm(A - P.T @ L @ U)
    #error = norm(P @ A - L @ U)

    # ------------------------------------------------------------
    # Número de condición (norma 2 por defecto)
    # ------------------------------------------------------------
    cond_A = np.linalg.cond(A)

    # ------------------------------------------------------------
    # Factor de crecimiento (indicador de estabilidad)
    # max |U_ij| / max |A_ij|
    # ------------------------------------------------------------
    growth = np.max(np.abs(U)) / np.max(np.abs(A))

    print(f"\nε = {eps}")
    print("A =\n", A)
    print("P =\n", P)
    print("L =\n", L)
    print("U =\n", U)
    print("Error ||PA - LU|| =", error)
    print("cond(A) =", cond_A)
    print("Factor de crecimiento =", growth)


# ============================================================
# Comparación con LU SIN pivoteo (solo con fines pedagógicos)
# ============================================================

def lu_sin_pivoteo(A):
    """
    Implementación simple de LU sin pivoteo (NO recomendada en práctica).
    """
    L = np.eye(2)
    U = A.copy().astype(float)

    L[1, 0] = U[1, 0] / U[0, 0]
    U[1, :] = U[1, :] - L[1, 0] * U[0, :]

    return L, U


print("\n===== Comparación: LU sin pivoteo (no usar en práctica) =====")

for eps in eps_values:
    A = np.array([[eps, 1.0],
                  [1.0, 1.0]])

    L, U = lu_sin_pivoteo(A)
    error = norm(A - L @ U)

    print(f"\nε = {eps}")
    print("L =\n", L)
    print("U =\n", U)
    print("Error ||A - LU|| =", error)


# =========================================================================
# Mensaje:
# =============================================================================
# - LU sin pivoteo puede ser numéricamente inestable (aunque la matriz A=LU sea bien condicionada!!)
# - El pivoteo parcial evita dividir por números pequeños.
# - En la práctica, es recomendable usar implementaciones robustas (SciPy/LAPACK).
# ============================================================================