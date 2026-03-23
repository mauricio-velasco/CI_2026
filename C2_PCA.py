"""
TUTORIAL: PCA desde cero con SVD
Proceso:
En R^2
- Generar datos gaussianos con covarianza conocida (oculta)
- Aplicar PCA usando SVD a partir de la muestra para descubrirla

Luego generar datos en mayor dimensión que son "esencialmente dos-dimensionales"
Buscar las mejores 3 dimensiones y en ese espacio ver la proyección de rango 2.
Entender PCA y ganar algo de intuición sobre el Teorema de Echart-Young
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Generación de datos
# -------------------------------

np.random.seed(42)

n = 500   # número de muestras
p = 2     # dimensión

# Definimos autovalores (varianzas principales)
lambdas = np.array([9.0, 1.0])  # varianza grande en una dirección, pequeña en otra

# Definimos una rotación (para "esconder" la estructura, teniendo datos más realistas)
theta = np.pi / 4  # 45 grados
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Construimos la covarianza que queremos tenga nuestra muestra:
Sigma = R @ np.diag(lambdas) @ R.T
print("Covarianza verdadera:\n", Sigma)

# Generamos datos gaussianos con la covarianza que queriamos
X = np.random.multivariate_normal(mean=[0, 0], cov=Sigma, size=n)
#Idea: Cada fila de X representa una medicion independientes de dos cantidades.
# --------------------------------------------
# 2. Visualización inicial de nuestros datos
# -------------------------------------------

plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.title("Datos generados")
plt.axis("equal")
plt.show()

#Estos son los datos iniciales con los que cuenta alguen experimental.
#El mecanismo de generación de datos es desconocido y queremos descubrirlo.

# -------------------------------
# 3. Centrado de los datos
# -------------------------------

# PCA requiere datos centrados
X_centered = X - X.mean(axis=0)

# -------------------------------
# 4. PCA:
# -------------------------------

# Descomposición SVD
# X = U S V^T
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Componentes principales (direcciones)
V = Vt.T

# -------------------------------
# 4. Proyección de los datos
# -------------------------------

# Coordenadas en la base PCA
Z = X_centered @ V

plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], alpha=0.3)
plt.title("Datos en coordenadas PCA (descorrelacionados)")
plt.axis("equal")
plt.show()

# -------------------------------
# 8. Reconstrucción con k componentes
# -------------------------------

k = 1  # usamos solo la primera componente

Z_k = Z[:, :k]
V_k = V[:, :k]

X_approx = Z_k @ V_k.T

plt.figure()
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.3, label="Original")
plt.scatter(X_approx[:, 0], X_approx[:, 1], alpha=0.3, label="Aprox (k=1)")
plt.legend()
plt.axis("equal")
plt.title("Aproximación de rango 1")
plt.show()

#PARTE 2: Datos en dimension mayor

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Datos en alta dimensión
# -------------------------------

np.random.seed(42)

n = 800   # muestras
p = 10    # dimensión alta

# Solo dos direcciones con varianza grande
lambdas = np.array([16.0, 9.0] + [0.5]*(p-2))

# Rotación aleatoria (matriz ortogonal)
Q, _ = np.linalg.qr(np.random.randn(p, p))

Sigma = Q @ np.diag(lambdas) @ Q.T

print("Dimensión:", p)
print("Autovalores verdaderos:", lambdas)

# Generación de datos
X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

# -------------------------------
# 2. Centrado
# -------------------------------

X_centered = X - X.mean(axis=0)

# -------------------------------
# 3. PCA via SVD
# -------------------------------

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T

# Varianza explicada
explained_variance_ratio = (S**2) / np.sum(S**2)
print("Varianza explicada (primeras 5):", explained_variance_ratio[:5])

# -------------------------------
# 4. Proyección
# -------------------------------

Z = X_centered @ V

# -------------------------------
# 5. Visualización 3D (primeras 3 PCs)
# -------------------------------

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], alpha=0.3)

ax.set_title("Proyección en las primeras 3 componentes principales")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()


# -------------------------------
# 6. Reconstrucción con k=2
# -------------------------------

k = 2

Z_k = Z[:, :k]
V_k = V[:, :k]

X_approx = Z_k @ V_k.T

# Error de reconstrucción
error = np.linalg.norm(X_centered - X_approx) / np.linalg.norm(X_centered)
print("Error relativo (k=2):", error)

# ------------------------------------------------
# Visualización de las componentes principales
# -----------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original points in PCA coordinates
x = Z[:, 0]
y = Z[:, 1]
z = Z[:, 2]

ax.scatter(x, y, z, alpha=0.2, label="Original (3D)")

# Projection onto first 2 PCs (set PC3 = 0)
z_proj = np.zeros_like(z)

ax.scatter(x, y, z_proj, alpha=0.4, label="Proyección (k=2)", color='red')

# Optional: draw vertical projection lines (nice but can be heavy)
for i in range(0, len(x), 20):  # subsample to avoid clutter
    ax.plot([x[i], x[i]], [y[i], y[i]], [z[i], 0], color='gray', alpha=0.2)

ax.set_title("Proyección sobre el plano PC1-PC2 (rango 2)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

ax.legend()
plt.show()

# Por el Teorema de Eckart-Young la aproximación de rango 2 que estamos visualizando 
# es la mejor posible (oara una cierta norma).