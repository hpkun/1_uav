"""Continuous moving-frame continuation of the paper's singular angle chart.

The continuation inside the polar cap and its bounded gauge relaxation are
engineering assumptions, NOT equations supplied by Xu. See docs/dynamics_v2.md.
Columns are (e_v,e_gamma,e_psi); this convention has determinant -1.
"""
import json
import math
import numpy as np


class SimulationFault(FloatingPointError):
    """A failed simulation transaction, never an air-combat terminal transition."""
    def __init__(self, reason, **context):
        self.diagnostic = dict(reason=reason, **context)
        super().__init__(json.dumps(self.diagnostic, allow_nan=True, default=lambda x: np.asarray(x).tolist()))


def initial_frame(state):
    g, p = state[4:6]
    sg, cg, sp, cp = math.sin(g), math.cos(g), math.sin(p), math.cos(p)
    return np.array([[cg*cp, -sg*cp, -sp], [cg*sp, -sg*sp, cp], [sg, cg, 0.]])


def frame_error(frame):
    return float(np.max(np.abs(frame.T @ frame - np.eye(3))))


def validate(state, frame):
    if (np.shape(state) != (6,) or np.shape(frame) != (3,3)
            or not np.isfinite(state).all() or not np.isfinite(frame).all()):
        raise SimulationFault('non-finite or malformed state/frame', state=state, frame=frame)
    if state[3] <= 0 or frame_error(frame) > 1e-8 or np.linalg.det(frame) > -.999999:
        raise SimulationFault('invalid speed or damaged orthonormal frame', state=state, frame=frame)
    if np.linalg.norm(initial_frame(state)[:,0]-frame[:,0]) > 1e-8:
        raise SimulationFault('state velocity and internal frame disagree', state=state, frame=frame)


def frame_rhs(value, control, config, diagnostics=None):
    """Position, speed, and frame derivatives; finite even at u=(0,0,+/-1)."""
    v = value[3]
    if not np.isfinite(value).all() or v <= 0:
        raise SimulationFault('invalid integration stage', stage=value, control=control)
    frame = value[4:].reshape(3,3)
    u, a, b = frame.T
    mu, zeta, normal = control
    # Full gravity projection is retained, including during gauge continuation.
    q = np.array([0.,0.,1.])-u[2]*u
    du = config.gravity/v*(normal*(math.cos(mu)*a+math.sin(mu)*b)-q)
    h = math.hypot(u[0],u[1])
    spin = 0.;delta=0.
    if h > 0:
        t = min(h/config.pole_cap_ratio, 1.)
        weight = t*t*(3-2*t)
        # q/h is the ordinary spherical pitch axis; the closest of its two
        # signs permits an unfolded pitch chart across a meridional pole.
        canonical_b = np.array([-u[1]/h,u[0]/h,0.])
        delta = math.atan2(float(q@b),float(q@a))
        # Smooth pi-periodic alignment: both unfolded spherical pitch charts
        # are valid. No discontinuous choice between their two reference axes.
        if h < config.pole_cap_ratio:
            # Compute w/h directly: never form an unbounded intermediate and
            # then multiply by zero (which would fail for subnormal h).
            connection = -u[2]*float(du@canonical_b)*(t*(3-2*t)/config.pole_cap_ratio)
        else:
            connection = -u[2]*float(du@canonical_b)/h
        spin = connection + weight*.5*config.frame_alignment_rate*math.sin(2*delta)
    if diagnostics is not None:
        diagnostics['max_frame_spin_rate']=max(diagnostics['max_frame_spin_rate'],abs(spin))
        diagnostics['max_reference_axis_error']=max(diagnostics['max_reference_axis_error'],
            abs(.5*math.atan2(math.sin(2*delta),math.cos(2*delta))))
    da = -float(du@a)*u + spin*b
    db = -float(du@b)*u - spin*a
    return np.r_[v*u, config.gravity*(zeta-u[2]), np.column_stack((du,da,db)).ravel()]


def _project(frame):
    error = frame_error(frame)
    if not np.isfinite(error) or error > .05:
        raise SimulationFault('frame integration failed orthogonality tolerance', error=error, frame=frame)
    u = frame[:,0]/np.linalg.norm(frame[:,0])
    a = frame[:,1]-float(frame[:,1]@u)*u
    length = np.linalg.norm(a)
    if length < .5:
        raise SimulationFault('collapsed local reference axis', frame=frame)
    a /= length
    return np.column_stack((u,a,np.cross(a,u))), error


def export_state(position, speed, frame, previous):
    """Choose the nearest equivalent UNFOLDED spherical chart, never a force axis.

    Heading at an exactly vertical direction is unobservable; retain its previous
    value there. Geometry uses the reconstructed physical velocity direction.
    """
    u = frame[:,0]
    h = math.hypot(u[0],u[1])
    pitch = math.atan2(u[2],h)
    yaw = math.atan2(u[1],u[0]) if h > 1e-14 else previous[5]
    candidates=[]
    for g,p in ((pitch,yaw),(math.pi-pitch,yaw+math.pi)):
        g += 2*math.pi*round((previous[4]-g)/(2*math.pi))
        p += 2*math.pi*round((previous[5]-p)/(2*math.pi))
        candidates.append((abs(g-previous[4])**2+abs(p-previous[5])**2,g,p))
    _,g,p = min(candidates)
    # Preserve the existing wrapped heading interface. The physical direction
    # and internal frame remain continuous across this conventional wrap.
    p = (p+math.pi)%(2*math.pi)-math.pi
    return np.r_[position,speed,g,p]


def step(state, control, frame, config):
    validate(state,frame)
    control=np.asarray(control,dtype=float)
    if (control.shape!=(3,) or not np.isfinite(control).all()
            or not -math.pi<=control[0]<=math.pi or not -1<=control[1]<=2 or not 0<=control[2]<=8):
        raise SimulationFault('invalid physical control', control=control)
    value=np.r_[state[:4],frame.ravel()]
    h0=math.hypot(*frame[:2,0]); minimum=h0; error=0.; entries=0; continuations=0
    # A fixed, disclosed near-cap resolution; no rejection of pole crossings.
    near = h0 < config.pole_cap_ratio + config.dt*config.gravity*(control[2]+1)/state[3]
    count=config.frame_polar_substeps if near else config.frame_normal_substeps
    dt=config.dt/count
    was_inside=h0<config.pole_cap_ratio
    rates=dict(max_frame_spin_rate=0.,max_reference_axis_error=0.)
    for _ in range(count):
        k1=frame_rhs(value,control,config,rates)
        k2=frame_rhs(value+dt*k1/2,control,config,rates)
        k3=frame_rhs(value+dt*k2/2,control,config,rates)
        k4=frame_rhs(value+dt*k3,control,config,rates)
        stage_min=min(math.hypot(*stage[4:].reshape(3,3)[:2,0]) for stage in
                      (value,value+dt*k1/2,value+dt*k2/2,value+dt*k3))
        minimum=min(minimum,stage_min)
        value += dt*(k1+2*k2+2*k3+k4)/6
        projected,err=_project(value[4:].reshape(3,3)); value[4:]=projected.ravel()
        h=math.hypot(*projected[:2,0]); minimum=min(minimum,h);error=max(error,err)
        inside=h<config.pole_cap_ratio
        entries+=int(inside and not was_inside); continuations+=int(inside or was_inside or stage_min<config.pole_cap_ratio)
        was_inside=inside
    clipped=float(np.clip(value[3],config.v_min,config.v_max))
    if not np.isfinite(value).all() or value[3]<=0:
        raise SimulationFault('non-finite or non-positive integration endpoint',endpoint=value,control=control)
    result=export_state(value[:3],clipped,projected,state)
    validate(result,projected)
    return result,projected,dict(min_horizontal_ratio=minimum,pole_entries=entries,
        continuation_substeps=continuations,frame_projection_error=error,
        frame_orthogonality_error=frame_error(projected),
        frame_norm_error=float(np.max(abs(np.linalg.norm(projected,axis=0)-1))),
        speed_clips=int(clipped!=value[3]),**rates)


def advance(state, control, frame, config):
    if config.integrator == 'frame_rk4':
        return step(state,control,frame,config)
    from env.integrator import rk4
    counters=dict(speed_clips=0,nonfinite_states=0,integration_failures=0); diagnostic={}
    result,ok=rk4(state,control,config,counters,diagnostics=diagnostic)
    if not ok:
        raise SimulationFault(diagnostic['reason'],state=state,control=control,integrator='angle_rk4')
    return result,initial_frame(result),dict(min_horizontal_ratio=abs(math.cos(result[4])),
        pole_entries=0,continuation_substeps=0,frame_projection_error=0.,frame_orthogonality_error=frame_error(initial_frame(result)),
        frame_norm_error=float(np.max(abs(np.linalg.norm(initial_frame(result),axis=0)-1))),
        speed_clips=counters['speed_clips'])
