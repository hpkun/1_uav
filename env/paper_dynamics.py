"""Standalone equation checks for Xu (2026), pp.2-3 Eqs.(1)-(2).

Shared by the runtime and standalone checks; no task or training logic. Gravity is caller supplied:
the paper names g but does not publish its numerical value. RK4, the singularity
threshold and failed-step handling below are disclosed numerical choices.
No gamma, speed, altitude or position projection is applied to the equations.
"""
from dataclasses import dataclass
import math

import numpy as np


class DynamicsDomainError(ValueError):
    """The selected speed/angle coordinates cannot be evaluated safely."""


def _finite_vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise DynamicsDomainError(f'{name} must be a finite {length}-vector')
    return result


def rhs(state, control, *, gravity, epsilon=1e-6):
    """SI state [x,y,z,v,gamma,psi], physical control [mu,zeta,zeta_hat].

    Table II (p.8) supplies control bounds. epsilon rejects singular inputs;
    it never replaces a denominator or limits the flight-path angle to 85 deg.
    """
    y = _finite_vector(state, 6, 'state')
    u = _finite_vector(control, 3, 'control')
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError('gravity must be finite and positive (engineering choice)')
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('epsilon must be finite and in (0,1)')
    v, gamma, psi = y[3:]
    mu, zeta, normal = u
    if not (-math.pi <= mu <= math.pi and -1 <= zeta <= 2 and 0 <= normal <= 8):
        raise DynamicsDomainError('control outside Table II')
    cg, sg = math.cos(gamma), math.sin(gamma)
    if v <= epsilon or abs(cg) <= epsilon:
        raise DynamicsDomainError('singular speed/flight-path-angle coordinates')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = np.array([
            v * cg * math.cos(psi), v * cg * math.sin(psi), v * sg,
            gravity * (zeta - sg),
            gravity / v * (normal * math.cos(mu) - cg),
            gravity * normal * math.sin(mu) / (v * cg),
        ])
    if not np.isfinite(result).all():
        raise DynamicsDomainError('nonfinite derivative')
    return result


@dataclass(frozen=True)
class EquationStep:
    state: np.ndarray
    accepted: bool
    reason: str
    within_published_speed_range: bool


def rk4_step(state, control, *, gravity, dt=0.25, epsilon=1e-6):
    """Unprojected classical RK4, with explicit rejection of unsafe stages.

    dt=.25 is published (p.8); RK4 is not. Crossing an angle-coordinate
    singularity between stage states also rejects the step. A rejected step
    returns the unchanged finite input, not an artificial finite derivative.
    Published speed-range membership is reported, never silently projected.
    """
    y = _finite_vector(state, 6, 'state').copy()
    u = _finite_vector(control, 3, 'control')
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')

    def evaluate(stage):
        # The pole-cell is bounded by pi/2 + k*pi; changing cell crosses a pole.
        if np.isfinite(stage).all() and math.floor((stage[4] + math.pi/2)/math.pi) != math.floor((y[4] + math.pi/2)/math.pi):
            raise DynamicsDomainError('RK4 stage crosses an angle-coordinate singularity')
        return rhs(stage, u, gravity=gravity, epsilon=epsilon)

    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            k1 = evaluate(y)
            k2 = evaluate(y + dt*k1/2)
            k3 = evaluate(y + dt*k2/2)
            k4 = evaluate(y + dt*k3)
            result = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        evaluate(result)  # finite/domain check of endpoint; no projection
    except DynamicsDomainError as exc:
        return EquationStep(y, False, str(exc), bool(100 <= y[3] <= 400))
    return EquationStep(result, True, '', bool(100 <= result[3] <= 400))
