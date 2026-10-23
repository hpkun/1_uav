"""Tables VI-X, transcribed in source order x,y,z,v,psi,mu,gamma (SI)."""
from dataclasses import dataclass, field
import numpy as np

TABLES = {
    'random_3v3': ('VI', 3, [
        [5000,4500,5800,210,1.20,.50,.15], [5000,6000,6200,205,-.80,-.30,-.20],
        [5000,7500,5500,215,.40,.80,.10], [10000,4300,7000,185,-2.50,.60,-.25],
        [10000,5800,6500,190,1.80,-.40,.30], [10000,7800,7200,180,-.30,.90,-.15]]),
    'head_on_3v3': ('VII', 3, [
        [5000,4900,6000,200,.01,.87,0], [5000,5900,6000,200,0,.34,0],
        [5000,6700,6000,200,-.01,-.32,0], [10000,5000,6000,200,3.04,-1.31,0],
        [10000,5900,6000,200,3.14,0,0], [10000,6200,6000,200,-3.14,1.31,0]]),
    'rear_end_3v3': ('VIII', 3, [
        [5000,4900,6000,200,3.06,-1.20,.25], [5000,5900,6000,200,-3.12,.60,-.30],
        [5000,6900,6000,200,3.10,-.10,.40], [8000,5000,6000,200,-3.04,.80,-.20],
        [8000,5900,6000,200,3.14,-.50,.35], [8000,7100,6000,200,3.07,.30,-.15]]),
    'representative_5v5': ('IX', 5, [
        [4600,4000,5300,220,1.80,-.80,.20], [5200,5200,6500,195,-.90,.60,-.30],
        [4800,6800,5800,230,.40,0,0], [5500,8200,7100,185,-1.60,-.50,.15],
        [4900,9300,5000,205,2.20,.40,-.10], [9800,3800,7800,175,-2.20,1.10,.20],
        [10500,5500,6200,210,1.40,-.70,-.15], [10000,7000,7500,190,-.50,0,.30],
        [10200,8600,5500,205,2.80,.50,-.10], [9500,9600,6800,180,-1.,-1.,.10]]),
    'representative_3v5': ('X', 3, [
        [5000,4200,5800,210,1.50,1.20,0], [5000,5600,6200,195,-2.20,-.80,0],
        [5000,7100,5500,220,.30,.50,0], [10000,4000,7000,185,2.80,-1.,0],
        [10000,5400,6500,205,-1.60,.60,0], [10000,6500,7200,190,.90,-.40,0],
        [10000,7800,6800,180,-2.50,.80,0], [10000,9000,7400,200,1.80,-.60,0]])}


def paper_scenario(name):
    table, n, rows = TABLES[name]
    raw = np.array(rows, dtype=np.float64)
    states = raw[:, [0,1,2,3,6,4]]  # mu is metadata, never a seventh state.
    return dict(friendly_states=states[:n].copy(), adversarial_states=states[n:].copy(),
                metadata=dict(table=table, initial_mu=raw[:,5].copy()))


@dataclass
class CombatState:
    friendly_states: np.ndarray
    adversarial_states: np.ndarray
    friendly_alive: np.ndarray
    adversarial_alive: np.ndarray
    assignments: np.ndarray
    friendly_dwell: np.ndarray
    adversarial_dwell: np.ndarray
    step_count: int = 0
    events: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    friendly_frames: np.ndarray = None
    adversarial_frames: np.ndarray = None


def generate(config, rng, options=None):
    options = dict(options or {})
    explicit = 'friendly_states' in options or 'adversarial_states' in options
    name = options.pop('scenario', config.initial_scenario)
    n, m = config.num_uavs, config.num_adversaries
    if not explicit and name != 'random_training':
        options = {**paper_scenario(name), **options}
        explicit = True
    if explicit:
        try:
            red = np.array(options['friendly_states'], dtype=np.float64, copy=True)
            blue = np.array(options['adversarial_states'], dtype=np.float64, copy=True)
        except KeyError as exc:
            raise ValueError('both teams require complete states') from exc
    else:
        # Disclosed minimal sampler, not the authors' unavailable distribution.
        states = []
        for k in range(n+m):
            for _ in range(10000):
                xyz = rng.uniform([4500 if k<n else 9500,3800,5000],
                                  [5500 if k<n else 10500,9600,7800])
                if not states or np.min(np.linalg.norm(np.array(states)[:,:3]-xyz, axis=1)) >= config.random_min_separation:
                    break
            else:
                raise ValueError('random minimum separation infeasible')
            states.append([*xyz, rng.uniform(175,230), rng.uniform(-.3,.3), rng.uniform(-np.pi,np.pi)])
        red, blue = np.array(states[:n]), np.array(states[n:])
    for a, shape in ((red,(n,6)),(blue,(m,6))):
        if a.shape != shape or not np.isfinite(a).all() or np.any((a[:,3]<config.v_min)|(a[:,3]>config.v_max)):
            raise ValueError('initial states require finite [N,6] and published speed range')
    alive = []
    for key, count in (('friendly_alive',n),('adversarial_alive',m)):
        mask = np.asarray(options.get(key, np.ones(count, dtype=bool)))
        if mask.shape != (count,) or not np.isin(mask, [0,1]).all():
            raise ValueError('initial alive masks must be binary stable-ID arrays')
        alive.append(mask.astype(bool))
    if not alive[0].any() or not alive[1].any():
        raise ValueError('reset requires at least one live UAV on each side')
    return CombatState(red, blue, *alive, np.full(n,-1,dtype=np.int64),
                       np.zeros((n,m),dtype=np.int64), np.zeros((m,n),dtype=np.int64),
                       metadata=options.get('metadata',{}))
