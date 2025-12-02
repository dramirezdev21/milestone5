# Milestone 5 - N Body Problem

import numpy as np
import matplotlib.pyplot as plt

from Cauchy_problem import Cauchy_problem
from Temporal_schemes import RK4  


# ---------------------------------------------------------
#  Generación de condiciones iniciales 
# ---------------------------------------------------------
def Initial_conditions(Nb, Nc):
    """
    Devuelve el vector de estado U0 :
    U (longitud 2*Nc*Nb) reorganizado como U[body, coord, pos/vel].
    """

    # U0 es un vector 1D que luego se reorganiza como (Nb, Nc, 2)
    U0 = np.zeros(2 * Nc * Nb)
    U1 = U0.reshape(Nb, Nc, 2)

    r0 = U1[:, :, 0]   # posiciones
    v0 = U1[:, :, 1]   # velocidades

    # -----------------------------------------------------
    # Sistema de 3 cuerpos 
    # -----------------------------------------------------

    # Masas (solo para referencia)
    # m = [1.0, 1e-3, 5e-4]

    # Body 0: "sol"
    r0[0, :] = [0.0, 0.0]
    v0[0, :] = [0.0, 0.0]

    # Body 1
    r0[1, :] = [1.0, 0.0]
    v0[1, :] = [0.0, np.sqrt(1.0 / 1.0)]  # v = sqrt(GM/r), con G=1

    # Body 2
    r0[2, :] = [1.8, 0.0]
    v0[2, :] = [0.0, np.sqrt(1.0 / 1.8)]

    return U0


# ---------------------------------------------------------
#  Ecuaciones del movimiento 
# ---------------------------------------------------------
def F_NBody(U, t, Nb, Nc, masses, G=1.0):
    """
    Ecuaciones dvi/dt = -G sum_j m_j (ri - rj)/|ri-rj|^3
    dr/dt = v
    """
    Us = U.reshape(Nb, Nc, 2)

    r = Us[:, :, 0]    # (Nb, Nc)
    v = Us[:, :, 1]    # (Nb, Nc)

    F = np.zeros_like(U)
    dUs = F.reshape(Nb, Nc, 2)

    drdt = dUs[:, :, 0]
    dvdt = dUs[:, :, 1]

    # dr_i/dt = v_i
    drdt[:, :] = v

    # dv_i/dt = sum_j!=i G*m_j*(r_j - r_i)/|r_j - r_i|^3
    for i in range(Nb):
        for j in range(Nb):
            if i != j:
                diff = r[j, :] - r[i, :]
                dist = np.linalg.norm(diff)
                dvdt[i, :] += G * masses[j] * diff / dist**3

    return F


# ---------------------------------------------------------
#  Integración principal 
# ---------------------------------------------------------
def Integrate_NBP():

    # Número de cuerpos y coordenadas
    Nb = 3
    Nc = 2   # (x,y)

    # masas de tu config. original
    masses = np.array([1.0, 1e-3, 5e-4])

    # tiempo
    N = 4000
    t0 = 0.0
    tf = 50.0
    Time = np.linspace(t0, tf, N + 1)

    # condiciones iniciales 
    U0 = Initial_conditions(Nb, Nc)

    # Definición del RHS como función anónima
    def F(U, t):
        return F_NBody(U, t, Nb, Nc, masses)

    # Integración usando RK4 
    U = Cauchy_problem(F, Time, U0, RK4)

    # Reorganizamos para graficar
    Us = U.reshape(N + 1, Nb, Nc, 2)
    R = Us[:, :, :, 0]      # posiciones -> (Nt, Nb, 2)

    # -----------------------------------------------------
    # Gráfico de trayectorias 
    # -----------------------------------------------------
    plt.figure(figsize=(7, 7))

    for i in range(Nb):
        plt.plot(R[:, i, 0], R[:, i, 1], label=f"Cuerpo {i}")
        plt.scatter(R[0, i, 0], R[0, i, 1], s=50)

    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.title("Órbitas del problema de 3 cuerpos ")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


# ---------------------------------------------------------
if __name__ == "__main__":
    Integrate_NBP()