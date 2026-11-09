"""Static reference geometry and time scaling, independent of physical state."""
from dataclasses import dataclass
import math

import numpy as np

from .interception.config import InterceptionConfig


VERSION = "uav-preset-trajectory-v1.0.0"
MODES = ("straight", "bezier")
DEFAULT_FLIGHT = InterceptionConfig()
_MAX_S1 = 15/8
_MAX_S2 = 10/math.sqrt(3)


@dataclass(frozen=True)
class TrajectoryConfig:
    starts: tuple = ((-22., -12., 15.),)
    endpoints: tuple = ((22., 12., 23.),)
    initial_velocities: tuple = ((.4, .2, 0.),)
    start_jitter: tuple = (3., 3., 2.)
    lower: tuple = DEFAULT_FLIGHT.lower
    upper: tuple = DEFAULT_FLIGHT.upper
    dt: float = DEFAULT_FLIGHT.dt
    tau: float = DEFAULT_FLIGHT.pursuer_tau
    vmax: float = DEFAULT_FLIGHT.pursuer_vmax
    horizontal_acceleration_max: float = DEFAULT_FLIGHT.horizontal_acceleration_max
    climb_vmax: float = DEFAULT_FLIGHT.climb_vmax
    descent_vmax: float = DEFAULT_FLIGHT.descent_vmax
    body_radius: float = DEFAULT_FLIGHT.pursuer_collision_radius
    event_time_tolerance: float = DEFAULT_FLIGHT.event_time_tolerance
    reference_speed: float = 6.
    bezier_lateral_range: tuple = (8., 14.)
    bezier_height_range: tuple = (3., 7.)
    length_order: int = 32
    position_gain: float = .7
    velocity_gain: float = .45
    hold_seconds: float = 6.
    max_steps: int = 1200
    position_tolerance: float = .25
    speed_tolerance: float = .1

    def __post_init__(self):
        for name in ("starts", "endpoints", "initial_velocities"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, tuple(map(tuple, a)))
        if not len(self.starts) == len(self.endpoints) == len(self.initial_velocities):
            raise ValueError("initial positions, velocities and endpoints must have the same length")
        for name in ("start_jitter", "lower", "upper", "bezier_lateral_range", "bezier_height_range"):
            a = tuple(getattr(self, name))
            if len(a) != (2 if name.endswith("range") else 3) or not all(map(math.isfinite, a)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, a)
        for name in ("dt", "tau", "vmax", "reference_speed", "horizontal_acceleration_max", "climb_vmax",
                     "descent_vmax", "body_radius", "event_time_tolerance", "position_gain", "velocity_gain",
                     "hold_seconds", "position_tolerance", "speed_tolerance"):
            x = getattr(self, name)
            if isinstance(x, bool) or not math.isfinite(x) or x <= 0:
                raise ValueError(f"invalid {name}")
        if self.reference_speed > self.vmax or not self.event_time_tolerance < self.dt/10:
            raise ValueError("invalid reference speed or event tolerance")
        if type(self.length_order) is not int or self.length_order < 4 or type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("invalid integration order or step budget")
        if min(self.start_jitter) < 0 or any(lo < 0 or hi < lo for lo, hi in
                                               (self.bezier_lateral_range, self.bezier_height_range)):
            raise ValueError("invalid geometry range")
        starts = np.asarray(self.starts)
        self.check_points(np.vstack((starts-self.start_jitter, starts+self.start_jitter, self.endpoints)))
        velocity = np.asarray(self.initial_velocities)
        if (np.linalg.norm(velocity,axis=1)>self.vmax).any() or (velocity[:,2]>self.climb_vmax).any() or (
                velocity[:,2]<-self.descent_vmax).any():
            raise ValueError("initial velocity violates flight limits")

    @property
    def flight_bounds(self):
        return np.asarray(self.lower)+self.body_radius, np.asarray(self.upper)-self.body_radius

    def check_points(self, points):
        low,high = self.flight_bounds
        if (low >= high).any() or not np.isfinite(points).all() or (points<low).any() or (points>high).any():
            raise ValueError("control point outside the legal flight envelope")


def _curve(points, u):
    """Bezier evaluation for a scalar parameter or an array of parameters."""
    if len(points) == 1:
        return np.broadcast_to(points[0], np.shape(u)+(3,)).copy()
    u = np.asarray(u)[...,None]
    if len(points) == 2:
        return (1-u)*points[0]+u*points[1]
    a = 1-u
    if len(points) == 3:
        return a*a*points[0]+2*a*u*points[1]+u*u*points[2]
    return a**3*points[0]+3*a*a*u*points[1]+3*a*u*u*points[2]+u**3*points[3]


@dataclass(frozen=True)
class ReferencePath:
    kind: str
    points: tuple
    length: float
    duration: float
    nominal_duration: float
    limiting_constraint: str
    certificate: tuple

    def sample(self, time):
        if not math.isfinite(time):
            raise ValueError("reference time must be finite")
        points = np.asarray(self.points)
        if time >= self.duration:
            return points[-1], np.zeros(3), np.zeros(3), "hold"
        if time <= 0.:
            return points[0], np.zeros(3), np.zeros(3), "tracking"
        r = time/self.duration
        small = min(r, 1-r)
        s = small**3*(10+small*(-15+6*small))
        u = s if r <= .5 else 1-s
        ud = 30*(r*(1-r))**2/self.duration
        udd = 60*r*(1-r)*(1-2*r)/self.duration**2
        first = (len(points)-1)*np.diff(points,axis=0)
        second = (len(first)-1)*np.diff(first,axis=0)
        tangent = _curve(first,u)
        acceleration = tangent*udd
        if len(second):
            acceleration += _curve(second,u)*ud**2
        return _curve(points,u), tangent*ud, acceleration, "tracking"


def build_path(kind, points, config):
    if kind not in MODES:
        raise ValueError("unsupported reference mode")
    points = np.asarray(points,dtype=float)
    if points.shape != (2 if kind=="straight" else 4,3):
        raise ValueError("invalid reference control points")
    config.check_points(points)
    first = (len(points)-1)*np.diff(points,axis=0)
    second = (len(first)-1)*np.diff(first,axis=0)
    if kind == "straight":
        length = float(np.hypot.reduce(first[0]))
    else:
        nodes,weights = np.polynomial.legendre.leggauss(config.length_order)
        length = float(weights@np.hypot.reduce(_curve(first,(nodes+1)/2),axis=1)/2)
    if not np.any(first):
        return ReferencePath(kind,tuple(map(tuple,points)),0.,0.,0.,"zero_length",())

    def bounds(coordinates):
        d1 = np.hypot.reduce(first[:,coordinates],axis=1).max()
        d2 = np.hypot.reduce(second[:,coordinates],axis=1).max() if len(second) else 0.
        return _MAX_S1*d1, _MAX_S1**2*d2+_MAX_S2*d1

    v,a = bounds(slice(None))
    _,ah = bounds(slice(0,2))
    _,az = bounds(slice(2,3))
    up = _MAX_S1*max(0.,first[:,2].max())
    down = _MAX_S1*max(0.,-first[:,2].min())
    c = config

    def command_time(speed_bound,acceleration_bound,limit):
        # Positive root of limit*T² - speed_bound*T - tau*acceleration_bound.
        return (speed_bound+math.sqrt(speed_bound**2+4*limit*c.tau*acceleration_bound))/(2*limit)

    nominal = length/c.reference_speed
    minimum = dict(control_period=c.dt,path_length=nominal, reference_speed=v/c.reference_speed,
                   horizontal_acceleration=math.sqrt(ah/c.horizontal_acceleration_max),
                   command_speed=command_time(v,a,c.vmax),
                   climb_command=command_time(up,az,c.climb_vmax),
                   descent_command=command_time(down,az,c.descent_vmax))
    constraint = max(minimum,key=minimum.get)
    t = max(minimum.values())
    certificate = dict(speed=v/t,horizontal_acceleration=ah/t**2,
                       command_speed=v/t+c.tau*a/t**2,
                       climb_command=up/t+c.tau*az/t**2,descent_command=down/t+c.tau*az/t**2)
    return ReferencePath(kind,tuple(map(tuple,points)),length,t,nominal,constraint,tuple(certificate.items()))


def initial_states(config, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed,0]))
    jitter = np.asarray(config.start_jitter)
    return np.column_stack((np.asarray(config.starts)+rng.uniform(-jitter,jitter,(len(config.starts),3)),
                            config.initial_velocities))


def make_paths(kind, states, config, seed):
    paths = []
    for i,(state,end) in enumerate(zip(states,config.endpoints,strict=True)):
        p0,p3 = state[:3],np.asarray(end)
        points = [p0,p3]
        if kind == "bezier":
            rng = np.random.default_rng(np.random.SeedSequence([seed,1,i]))
            delta = p3-p0
            if np.any(delta):
                side = np.cross([0.,0.,1.],delta)
                norm = np.linalg.norm(side)
                side = side/norm if norm else np.array([1.,0.,0.])
                sign = rng.choice([-1.,1.])
                bend = sign*rng.uniform(*config.bezier_lateral_range,(2,1))*side
                bend[:,2] = rng.uniform(*config.bezier_height_range,2)
                points = [p0,p0+delta/3+bend[0],p0+2*delta/3+bend[1],p3]
            else:
                points = [p0]*4
        paths.append(build_path(kind,points,config))
    return paths
