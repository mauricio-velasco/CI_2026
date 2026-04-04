import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================================================
# Parte 1: Visualizar bolas de p-norma en dimension 2,3
# ======================================================

def plot_p_norm_ball_2d(p_list, n_points=500):
    """
    Dibuja bolas unitarias de p-norma en 2D para los valores de p dados.
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    
    plt.figure(figsize=(8,8))
    for p in p_list:
        # Transformación para aproximar el círculo de p-norma
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/p)
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/p)
        plt.plot(x, y, label=f'p={p}')
    
    plt.title("Bolas unitarias de p-norma en 2D")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_p_norm_ball_3d(p_list, n_points=100):
    """
    Dibuja bolas unitarias de p-norma en 3D para los valores de p dados.
    """
    u = np.linspace(0, np.pi, n_points)
    v = np.linspace(0, 2*np.pi, n_points)
    U, V = np.meshgrid(u,v)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    for p in p_list:
        X = np.sign(np.sin(U)*np.cos(V)) * np.abs(np.sin(U)*np.cos(V))**(2/p)
        Y = np.sign(np.sin(U)*np.sin(V)) * np.abs(np.sin(U)*np.sin(V))**(2/p)
        Z = np.sign(np.cos(U)) * np.abs(np.cos(U))**(2/p)
        ax.plot_surface(X, Y, Z, alpha=0.3)
    
    ax.set_title("Bolas unitarias de p-norma en 3D")
    plt.show()


# Ejercicio 1:
# Justifique el código de arriba. 
# Demuestre que permite dibujar las bolas unitarias de las p-normas (notar son normas para p>=1 solamente)
p_list = [0.5, 0.7,1.0, 1.5, 2.0, 2.5, 3.0, 10]
plot_p_norm_ball_2d(p_list)

# Ejercicio 2:
#En $3D$, solo dos
p_list = [1, 2.0,3.0]
plot_p_norm_ball_3d(p_list)  # Opcional: bolas en 3D

# ==================================================================
# Parte 2: Demostrar comportamiento de sensibilidad al valor máximo
# ==================================================================

def show_max_sensitive_behavior(dim=5, n_samples=1000, p_list=[1,2,5,10,20,np.inf]):
    """
    Genera vectores aleatorios y compara cómo las p-normas miden su magnitud.
    Muestra que normas con p grandes son dominadas por los componentes más grandes.
    """
    # Generar vectores aleatorios (por ejemplo, vectores de error)
    # con vectores normales standard
    vectores = np.random.randn(n_samples, dim)
    norm_values = {p: [] for p in p_list}
    
    for p in p_list:
        if p == np.inf:
            norm_values[p] = np.max(np.abs(vectores), axis=1)
        else:
            norm_values[p] = np.linalg.norm(vectores, ord=p, axis=1)
    
    # Diagrama de caja para visualizar la distribución de normas
    plt.figure(figsize=(10,6))
    data = [norm_values[p] for p in p_list]
    plt.boxplot(data, labels=[str(p) for p in p_list])
    plt.title(f"Distribución de p-normas para vectores de dimensión {dim}")
    plt.ylabel("Valor de la norma")
    plt.xlabel("p")
    plt.grid(True)
    plt.show()
    
    # Opcional: mostrar correlación con el componente máximo
    max_componentes = np.max(np.abs(vectores), axis=1)
    print("Correlación del componente máximo con varias p-normas:")
    for p in p_list:
        corr = np.corrcoef(max_componentes, norm_values[p])[0,1]
        print(f"  p={p:>2}: correlación = {corr:.2f}")

# Ejercicio: Explique la siguiente tabla de observaciones
# usando su entendimiento de las p-normas. Por qué son así los boxplots?
# los datos estan en la bola euclidea de dimensión 5.
# (quizás aclara re-hacer con datos unitarios?)
p_list = [1.0, 1.5, 2.0, 2.5, 3.0, 10,50]
show_max_sensitive_behavior(p_list= p_list)


# ==================================================================
# Parte 3: Normas matriciales
# ==================================================================


A = np.array([[1,0,0,0,0],
              [0,1,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0]])

#Ejercicio 1: 
# (1) Calcule la Schatten p-norm de A para todo $p>=1$

B = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# Ejercicio 2: 
# Escriba un script que:
# Dado p>=1 calcule la Schatten p-norm de B.
# Calcule la norma inducida (1,1), (2,2) e (inf,inf) de B
# Como estimaría la norma inducida (2,3) de B?

# Ejericio 3: 
# Acotar la norma inducida de A para cualquier norma inducida
#(sugerencia: Si A es de rango 1, cuánto es su norma inducida?)
# **Concluya que las Schatten norms no son inducidas casi nunca.

