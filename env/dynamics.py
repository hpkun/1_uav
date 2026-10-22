"""Historical Coop3D dynamics: Xu equations plus legacy, non-paper gamma clipping.

For standalone unmodified-equation checks use env.paper_dynamics; this module's
85-degree/denominator projections are retained for historical model compatibility.
"""
import numpy as np


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
    _, _, _, v, gamma, psi = state
    mu, zeta, normal = control
    # Protect all intermediate RK4 evaluations, including near-zero speed.
    v_safe = max(abs(v), config.denominator_epsilon)
    gamma = np.clip(gamma, -config.gamma_limit, config.gamma_limit)
    cg, sg = np.cos(gamma), np.sin(gamma)
    cos_safe = max(cg, config.denominator_epsilon)
    g = config.gravity
    return np.array([v * cg * np.cos(psi), v * cg * np.sin(psi), v * sg,
                     g * (zeta - sg), g / v_safe * (normal * np.cos(mu) - cg),
                     g * normal * np.sin(mu) / (v_safe * cos_safe)])
