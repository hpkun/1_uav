"""Published constants and explicitly separate reproduction assumptions."""
from dataclasses import asdict, dataclass
import math

ENVIRONMENT_VERSION = 'xu-aircombat-v1.0.0'
SCALES = {'3v3': (3, 3), '5v5': (5, 5), '3v5': (3, 5)}


@dataclass(frozen=True)
class AirCombatConfig:
    scale: str = '3v3'
    dt: float = .25
    max_steps: int = 250
    tau_max: float = 2000.
    tau_min: float = 100.
    alpha_max: float = math.pi / 4
    v_min: float = 100.
    v_max: float = 400.
    attack_eta: float = 5.
    terminal_e1: float = 20.
    terminal_e2: float = 25.
    attack_dwell_steps: int = 2
    opponent_best_probability: float = .8
    assignment_mode: str = 'aph'
    allied_penalty_enabled: bool = True
    opponent_aggregation: str = 'aggregate-sum'
    # Unpublished engineering choices; NOT additional paper constants.
    gravity: float = 9.81
    denominator_epsilon: float = 1e-6
    geometry_epsilon: float = 1e-9
    shootdown_r0: float = 10.  # PAPER DOES NOT REPORT r0; smoke placeholder.
    initial_scenario: str = 'random_training'
    random_min_separation: float = 100.

    def __post_init__(self):
        if self.scale not in SCALES:
            raise ValueError('scale must be 3v3, 5v5 or 3v5')
        if self.assignment_mode not in ('aph','nearest'):
            raise ValueError('invalid assignment_mode')
        if type(self.allied_penalty_enabled) is not bool:
            raise ValueError('invalid allied_penalty_enabled')
        if self.opponent_aggregation not in ('aggregate-sum','opportunistic-max','threat-aware-maximin'):
            raise ValueError('invalid opponent_aggregation')
        for key in ('dt', 'tau_max', 'tau_min', 'alpha_max', 'v_min', 'v_max',
                    'gravity', 'denominator_epsilon', 'geometry_epsilon', 'terminal_e2'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'invalid {key}')
        for key in ('attack_eta', 'terminal_e1', 'shootdown_r0', 'random_min_separation'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'invalid {key}')
        for key in ('max_steps', 'attack_dwell_steps'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'invalid {key}')
        if self.v_min >= self.v_max or self.tau_min >= self.tau_max:
            raise ValueError('invalid speed/distance intervals')
        if not 0 <= self.opponent_best_probability <= 1 or self.alpha_max > math.pi:
            raise ValueError('invalid angle/probability')

    @property
    def num_uavs(self):
        return SCALES[self.scale][0]

    @property
    def num_adversaries(self):
        return SCALES[self.scale][1]

    def to_dict(self):
        return asdict(self)

    def reproduction_assumptions(self):
        return dict(
            gravity=self.gravity, integrator='classical RK4; reject singular stages/crossings',
            numerical_failure='rollback whole joint step; end episode without combat death or reward',
            endpoint_projection='speed clipped to [100,400]; heading wrapped; no gamma/height clipping',
            geometry_epsilon=self.geometry_epsilon, zero_distance='zero LOS cosines; no attack; -1/epsilon posture',
            shootdown_r0=dict(value=self.shootdown_r0, source='论文未报告; experimental placeholder'),
            observations='raw SI features; live unassigned flag=1 and own v/gamma/psi/z, other fields zero',
            critic='five stable friendly 12D slots, zero padded; shared trunk/five value heads',
            simultaneous_events='post-motion snapshot; lowest attacker ID owns each victim; mutual kills allowed',
            dwell='all live pairs, independent of assignment; two inclusive full post-step tests; no reset test',
            reward_timing='assigned pre-action target, post-motion pre-death attack/posture; allied post-death survivors',
            terminal='pay current-mask agents including newly killed; never past dead; mutual elimination counts all-Blue-eliminated win',
            random_training=dict(x_red=[4500,5500], x_blue=[9500,10500], y=[3800,9600],
                                 z=[5000,7800], v=[175,230], gamma=[-.3,.3], psi=[-math.pi,math.pi],
                                 distribution='independent uniform; reject positions closer than minimum across both teams',
                                 minimum_separation=self.random_min_separation, rejection_limit=10000),
            ties='lowest stable ID / earliest Table I action',
            candidate_failure='score -infinity; if fewer than two feasible Blue actions abort numerical episode',
            reset_rng='initial seed 1000*s+r; subsequent resets continue worker Generator stream')
