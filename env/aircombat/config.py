"""Published constants and explicitly separate reproduction assumptions."""
from dataclasses import asdict, dataclass
import math

ENVIRONMENT_VERSION = 'xu-aircombat-v2.0.0'
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
    integrator: str = 'frame_rk4'
    pole_cap_ratio: float = .1
    frame_alignment_rate: float = 2.
    frame_normal_substeps: int = 1
    frame_polar_substeps: int = 8
    posture_reward_mode: str = 'paper'
    posture_gate_speed: float = 100.
    # The existing reward selector retains its historical name for checkpoints.
    madsac_battlefield_radius: float = 8000.
    madsac_distance_threshold: float = 4000.
    madsac_event_reward: float = 10.
    madsac_exit_penalty: float = 10.
    madsac_guidance_reward: float = .001
    madsac_aspect_limit_deg: float = 30.
    madsac_angle_limits_deg: tuple = (30.,15.,5.)
    madsac_offense_rewards: tuple = (.01,.02,.10)
    madsac_defense_penalties: tuple = (.015,.025,.15)

    def __post_init__(self):
        if self.posture_reward_mode not in ('paper','approach_gated','madsac_inspired'):
            raise ValueError('invalid posture_reward_mode')
        for key in ('madsac_battlefield_radius','madsac_distance_threshold','madsac_event_reward',
                    'madsac_exit_penalty','madsac_guidance_reward','madsac_aspect_limit_deg'):
            value=getattr(self,key)
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'invalid {key}')
        for key in ('madsac_angle_limits_deg','madsac_offense_rewards','madsac_defense_penalties'):
            values=tuple(getattr(self,key))
            if len(values)!=3 or any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values):
                raise ValueError(f'invalid {key}')
            object.__setattr__(self,key,values)  # JSON lists and checkpoint tuples have one identity.
        if not (180>=self.madsac_aspect_limit_deg>0 and 90>=self.madsac_angle_limits_deg[0]
                >self.madsac_angle_limits_deg[1]>self.madsac_angle_limits_deg[2]):
            raise ValueError('invalid MADSAC angle intervals')
        for values in (self.madsac_offense_rewards,self.madsac_defense_penalties):
            if not values[0]<values[1]<values[2]:raise ValueError('invalid MADSAC reward tiers')
        if isinstance(self.posture_gate_speed,bool) or not math.isfinite(self.posture_gate_speed) or self.posture_gate_speed<=0:
            raise ValueError('invalid posture_gate_speed')
        if self.integrator not in ('frame_rk4','angle_rk4'):
            raise ValueError('invalid integrator')
        if not 0 < self.pole_cap_ratio < 1 or not math.isfinite(self.frame_alignment_rate) or self.frame_alignment_rate <= 0:
            raise ValueError('invalid frame continuation parameters')
        for key in ('frame_normal_substeps','frame_polar_substeps'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'invalid {key}')
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
            initial_scenario=self.initial_scenario,
            head_on='random_training xyz/v and rejection; gamma=0, Red psi=0, Blue psi=pi; reserve legacy angle RNG draws',
            madsac_inspired=dict(source='Li et al. 2023 Eq.(25); independent reward, no Xu terms',
                geometry='Fig.2: ATA own horizontal velocity to LOS; AA target horizontal velocity to same LOS; HA LOS elevation to horizontal plane',
                equation6_ambiguity='printed ATA uses -V2 and reverses LOS, duplicating AA; use Fig.2 own-velocity ATA',
                degeneracy='zero horizontal LOS/own velocity: no ATA process reward; zero target horizontal velocity: no AA tier; finite sentinel angles; degree comparisons tolerate 1e-12 roundoff',
                timing='post-motion, post-death survivors for process rewards; pre-action APH target; R1 all actual simultaneous events',
                adaptation='R41+worst R42 over all surviving Blue; inclusive 4000m allows R3 and R4 together',
                battlefield='reward-only horizontal circle about mean of ALL initial positions; inside-to-outside transitions only',
                battlefield_radius=self.madsac_battlefield_radius),
            posture_reward_mode=self.posture_reward_mode,posture_gate_speed=self.posture_gate_speed,
            posture_gate='reward-only experimental ablation; pre-action assigned target; min(self LOS motion, relative closure)',
            gravity=self.gravity, integrator=self.integrator,
            continuation='engineering assumption: smooth polar connection cutoff and bounded closest-chart gauge relaxation',
            pole_cap_ratio=self.pole_cap_ratio,frame_alignment_rate=self.frame_alignment_rate,
            frame_substeps=[self.frame_normal_substeps,self.frame_polar_substeps],
            numerical_failure='explicit SimulationFault; no transition, terminal reward, automatic reset or PPO sample',
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
            candidate_failure='frame mode: explicit fault; angle baseline: infeasible candidate -inf, fewer than two abort; pure copies',
            reset_rng='initial seed 1000*s+r; subsequent resets continue worker Generator stream')
