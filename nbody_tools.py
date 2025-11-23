# nbody_tools.py
#
# Utilities for the N-body gravitational problem (2D, planar).
# - make_nbody_rhs: builds F(U, t) for given masses
# - total_energy: computes kinetic + potential energy
# - reshape_solution: convenience function for plotting

import numpy as np


def make_nbody_rhs(masses, G=1.0, softening=1e-3):
    """
    Build the RHS F(U, t) for the planar N-body problem.

    State vector U has the structure:
        U = [x1, y1, ..., xN, yN, vx1, vy1, ..., vxN, vyN]

    Parameters
    ----------
    masses : array_like of shape (N,)
        Mass of each body.
    G : float, optional
        Gravitational constant (default 1.0 in normalized units).
    softening : float, optional
        Softening length to avoid singularities when particles get very close.

    Returns
    -------
    F : callable
        Function F(U, t) to be used in the Cauchy problem solver.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)

    def F(U, t):
        # Split state into positions and velocities
        pos = U[:2 * N].reshape(N, 2)
        vel = U[2 * N:].reshape(N, 2)

        # Initialize accelerations to zero
        acc = np.zeros_like(pos)

        # Compute pairwise gravitational accelerations
        for i in range(N):
            # Vector from i to all j
            diff = pos - pos[i]               # shape (N, 2)
            dist2 = np.sum(diff**2, axis=1)   # squared distance
            dist2 += softening**2             # softening to avoid 1/0
            dist3 = dist2**1.5

            # Contribution of each j to the acceleration of body i
            contrib = G * masses[:, None] * diff / dist3[:, None]
            contrib[i] = 0.0  # no self-interaction
            acc[i] = contrib.sum(axis=0)

        # dU/dt = [v, a]
        dpos_dt = vel
        dvel_dt = acc
        return np.concatenate([dpos_dt.ravel(), dvel_dt.ravel()])

    return F


def total_energy(U, masses, G=1.0):
    """
    Compute total energy (kinetic + potential) of the system.

    Parameters
    ----------
    U : ndarray
        State vector at a given time.
    masses : array_like
        Masses of bodies.
    G : float
        Gravitational constant.

    Returns
    -------
    E : float
        Total mechanical energy.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)

    pos = U[:2 * N].reshape(N, 2)
    vel = U[2 * N:].reshape(N, 2)

    # Kinetic: 1/2 m v^2
    kinetic = 0.5 * np.sum(masses[:, None] * (vel**2))

    # Potential: - G sum_{i<j} m_i m_j / r_ij
    potential = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = pos[j] - pos[i]
            dist = np.linalg.norm(r_ij)
            if dist > 0.0:
                potential -= G * masses[i] * masses[j] / dist

    return kinetic + potential


def reshape_solution(U, masses):
    """
    Helper to reshape a full trajectory U(t) for plotting.

    Parameters
    ----------
    U : ndarray of shape (Nt, 4N)
        Full time history of the state vector.
    masses : array_like of length N

    Returns
    -------
    positions : ndarray of shape (Nt, N, 2)
        Positions over time for each body.
    velocities : ndarray of shape (Nt, N, 2)
        Velocities over time.
    """
    masses = np.array(masses, dtype=float)
    N = len(masses)

    pos = U[:, :2 * N].reshape(-1, N, 2)
    vel = U[:, 2 * N:].reshape(-1, N, 2)
    return pos, vel
