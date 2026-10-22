"""Xu Eqs.(1)-(2); the sole RHS implementation is paper_dynamics.rhs."""
import numpy as np
from .paper_dynamics import rhs


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def physical_actions(actions):
    """Map [0,1]^3 to [mu,zeta,zeta_hat]."""
    a = np.asarray(actions, dtype=np.float64)
    return a * np.array([2 * np.pi, 3.0, 8.0]) + np.array([-np.pi, -1.0, 0.0])


def velocity_vector(state):
    v, gamma, psi = np.moveaxis(np.asarray(state)[..., 3:6], -1, 0)
    return np.stack((v * np.cos(gamma) * np.cos(psi), v * np.cos(gamma) * np.sin(psi), v * np.sin(gamma)), axis=-1)


def derivatives(state, control, config):
    return rhs(state, control, gravity=config.gravity, epsilon=config.denominator_epsilon)
