# Milestone5.py
#
# Milestone 5: N-body problem
# ---------------------------
# 1) Define and integrate the planar N-body gravitational problem.
# 2) Simulate a simple 3-body configuration and visualize the result.

import numpy as np
import matplotlib.pyplot as plt

from Cauchy_problem import Cauchy_problem
from Temporal_schemes import RK4   # we reuse last year's RK4 implementation
from nbody_tools import make_nbody_rhs, total_energy, reshape_solution


def run_three_body_example():
    """
    Three-body planar gravitational system:
    - Body 0: "star" at the origin, massive and nearly fixed.
    - Body 1: "planet" on a roughly circular orbit.
    - Body 2: lighter body on a wider orbit.

    All units are non-dimensional.
    """

    # Gravitational constant
    G = 1.0

    # Masses: one heavy, two light
    masses = np.array([1.0, 1e-3, 5e-4])

    N_bodies = len(masses)

    # --- Initial positions (x, y) ---
    # Star at origin
    x0, y0 = 0.0, 0.0

    # Planet 1 at radius r1
    r1 = 1.0
    x1, y1 = r1, 0.0

    # Planet 2 at radius r2 > r1
    r2 = 1.8
    x2, y2 = r2, 0.0

    # --- Initial velocities (vx, vy) ---
    # Approximate circular velocities (central field)
    v1 = np.sqrt(G * masses[0] / r1)
    v2 = np.sqrt(G * masses[0] / r2)

    # Star: small velocity so total momentum is ~0
    vx0, vy0 = 0.0, 0.0

    # Planet 1: perpendicular to radius (along +y)
    vx1, vy1 = 0.0, v1

    # Planet 2: perpendicular to radius (along +y)
    vx2, vy2 = 0.0, v2

    # Pack initial state vector U0 = [x, y, vx, vy] for all bodies
    U0 = np.array([x0, y0, x1, y1, x2, y2, vx0, vy0, vx1, vy1, vx2, vy2],
                  dtype=float)

    # Time grid
    t0 = 0.0
    tf = 50.0      # total time
    Nt = 5000      # number of steps
    t = np.linspace(t0, tf, Nt + 1)

    # Build the RHS for this N-body system
    F = make_nbody_rhs(masses, G=G, softening=1e-3)

    # Integrate using the generic Cauchy problem solver + RK4
    U = Cauchy_problem(F, t, U0, Temporal_scheme=RK4)

    # Reshape solution for easier plotting
    positions, velocities = reshape_solution(U, masses)

    # -------------------------------------------------------
    # Plot trajectories in configuration space
    # -------------------------------------------------------
    colors = ["gold", "tab:blue", "tab:orange"]

    plt.figure(figsize=(7, 7))
    for k in range(N_bodies):
        plt.plot(positions[:, k, 0], positions[:, k, 1],
                 label=f"Body {k}", color=colors[k])
        plt.scatter(positions[0, k, 0], positions[0, k, 1],
                    marker="o", color=colors[k])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Three-body gravitational problem (planar)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # Plot total energy vs time (diagnostic of conservation)
    # -------------------------------------------------------
    energy = np.zeros(len(t))
    for i in range(len(t)):
        energy[i] = total_energy(U[i, :], masses, G=G)

    plt.figure(figsize=(8, 4))
    plt.plot(t, energy)
    plt.xlabel("t")
    plt.ylabel("Total energy")
    plt.title("Energy evolution in the three-body simulation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_three_body_example()
