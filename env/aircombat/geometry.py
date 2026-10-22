"""Xu Eqs.(3)-(6): full 3D geometry; alpha_j uses the SAME i->j LOS."""
from dataclasses import dataclass
import numpy as np
from env.dynamics import velocity_vector


@dataclass(frozen=True)
class Geometry:
    distance: np.ndarray
    cos_alpha_i: np.ndarray
    cos_alpha_j: np.ndarray
    cos_theta: np.ndarray
    alpha_i: np.ndarray
    alpha_j: np.ndarray


def relative_geometry(attackers, targets, epsilon=1e-9):
    a, b = np.asarray(attackers, dtype=float), np.asarray(targets, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 6 or b.shape[1] != 6 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('geometry requires finite [N,6], [M,6] states')
    los = b[None,:,:3]-a[:,None,:3]
    # hypot evaluates the same Euclidean norm with less overflow/roundoff than
    # squaring and summing (notably on the closed 45-degree cone boundary).
    d = np.hypot.reduce(los,axis=-1)
    va, vb = velocity_vector(a), velocity_vector(b)
    ua = va/np.maximum(np.linalg.norm(va,axis=-1,keepdims=True),epsilon)
    ub = vb/np.maximum(np.linalg.norm(vb,axis=-1,keepdims=True),epsilon)
    direction = los/np.maximum(d[...,None],epsilon)
    ci = np.clip(np.einsum('ik,ijk->ij', ua, direction), -1, 1)
    cj = np.clip(np.einsum('jk,ijk->ij', ub, direction), -1, 1)
    # atan2(||u x r||, u dot r) is exactly the same unsigned 3D angle as
    # acos(dot/(norms)), but avoids acos amplifying a one-ULP norm error at
    # closed cone boundaries. No angular tolerance or cone enlargement.
    ai = np.arctan2(np.linalg.norm(np.cross(ua[:,None,:],los),axis=-1),
                    np.einsum('ik,ijk->ij',ua,los))
    aj = np.arctan2(np.linalg.norm(np.cross(ub[None,:,:],los),axis=-1),
                    np.einsum('jk,ijk->ij',ub,los))
    # Zero LOS has no direction: explicitly use pi/2 (cos=0), and never fire.
    ai = np.where(d>epsilon,ai,np.pi/2)
    aj = np.where(d>epsilon,aj,np.pi/2)
    return Geometry(d,ci,cj,np.clip(ua@ub.T,-1,1),ai,aj)


def attack_advantage(g, config):
    # Closed cone/range boundaries; strict alpha_j < pi/2. Zero LOS cannot fire.
    inside = ((g.distance > config.geometry_epsilon) & (g.distance <= config.tau_max)
              & (g.alpha_i <= config.alpha_max) & (g.alpha_j < np.pi/2))
    return config.attack_eta*inside


def posture_advantage(g, attacker_states, config):
    d = g.distance
    close = -1/np.maximum(d,config.geometry_epsilon)
    middle = g.cos_alpha_i+g.cos_alpha_j
    speed = (np.asarray(attacker_states)[:,3,None]-config.v_min)/(config.v_max-config.v_min)
    far = speed*(g.cos_alpha_i+np.maximum(0,g.cos_theta))
    return np.where(d<config.tau_min, close, np.where(d<=config.tau_max,middle,far))


def pair_advantages(red, blue, config):
    g = relative_geometry(red,blue,config.geometry_epsilon)
    reverse = relative_geometry(blue,red,config.geometry_epsilon)
    return g, attack_advantage(g,config), attack_advantage(reverse,config), posture_advantage(g,red,config)
