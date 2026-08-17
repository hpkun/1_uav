"""Authoritative synchronous 4-red-vs-4-blue paper environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..config import aircraft_spec, load_config
from ..controller import TargetStateController
from ..dynamics import PointMassDynamics
from ..integrator import RK4Integrator
from ..models import AircraftState
from .death import DeathCause, death_summary
from .fixed_policy import NearestTargetPursuitPolicy
from .geometry import compute_paper_geometry
from .observation import OBSERVATION_DIM, build_team_observations
from .reward import equation25_event_reward, equation25_geometric_reward
from .scenario import random_diameter_states
from .sensor import SensorModel
from .weapon import WeaponModel


class PaperUAVCombatEnv:
    team_size, observation_dim, action_dim = 4, OBSERVATION_DIM, 3

    def __init__(self, config: str | Path | dict[str, Any] = "configs/paper_environment.yaml", sensor_noise: bool | None = None) -> None:
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.spec = aircraft_spec(self.config)
        self.dt = float(self.config["simulation"]["dt"]); self.max_steps = int(self.config["simulation"]["max_steps"])
        self.radius = float(self.config["battlefield"]["diameter"]) / 2.0
        assumptions = self.config["reproduction_assumptions"]
        sensor_cfg = dict(assumptions["sensor"]); formal_enabled = bool(sensor_cfg.pop("enabled"))
        self.sensor_noise_test_override = sensor_noise is not None
        self.sensor = SensorModel(**sensor_cfg, enabled=formal_enabled if sensor_noise is None else sensor_noise)
        self.weapon = WeaponModel(**(self.config["weapon"] | assumptions["weapon"]))
        action_cfg = self.config["action"]
        self.controller = TargetStateController(
            delta_yaw_max=max(map(abs, action_cfg["delta_psi"])),
            delta_pitch_max=max(map(abs, action_cfg["delta_theta"])),
            delta_speed_max=max(map(abs, action_cfg["delta_v"])),
        )
        self.dynamics, self.integrator = PointMassDynamics(), RK4Integrator(self.dt)
        self.fixed_policy = NearestTargetPursuitPolicy(
            delta_psi_max=self.controller.delta_yaw_max,
            delta_theta_max=self.controller.delta_pitch_max,
            delta_v_max=self.controller.delta_speed_max,
            desired_speed=assumptions["fixed_policy_desired_speed"],
        )
        self.scenario_cfg, self.obs_cfg = assumptions["formation"], assumptions["observation_normalization"]
        self.rng = np.random.default_rng(); self.red: list[AircraftState] = []; self.blue: list[AircraftState] = []
        self.red_phi, self.blue_phi = np.zeros(4), np.zeros(4)
        self.red_death_causes = np.zeros(4, np.int8); self.blue_death_causes = np.zeros(4, np.int8)
        self.steps = 0; self.red_attack_kills = 0; self.blue_attack_kills = 0
        self.last_red_observations = np.zeros((4,45),np.float32); self.last_blue_observations = self.last_red_observations.copy()

    @property
    def red_alive_mask(self) -> np.ndarray:
        return np.asarray([s.alive for s in self.red], dtype=np.float32)

    @property
    def blue_alive_mask(self) -> np.ndarray:
        return np.asarray([s.alive for s in self.blue], dtype=np.float32)

    @property
    def red_boundary_losses(self) -> int:
        return int(np.count_nonzero(self.red_death_causes == DeathCause.BOUNDARY))

    @property
    def blue_boundary_losses(self) -> int:
        return int(np.count_nonzero(self.blue_death_causes == DeathCause.BOUNDARY))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.red, self.blue, diameter_angle = random_diameter_states(self.rng, **self.scenario_cfg)
        self.red_phi[:] = 0; self.blue_phi[:] = 0; self.red_death_causes[:] = DeathCause.NONE; self.blue_death_causes[:] = DeathCause.NONE
        self.steps = self.red_attack_kills = self.blue_attack_kills = 0
        red_obs, blue_obs = self._observations_both()
        return red_obs, {"diameter_angle": diameter_angle, "red_ids": [f"red_{i}" for i in range(4)], "blue_ids": [f"blue_{i}" for i in range(4)], "red_observations": red_obs, "blue_observations": blue_obs, "red_alive_mask": self.red_alive_mask, "blue_alive_mask": self.blue_alive_mask}

    def _observed(self) -> tuple[list, list]:
        return ([self.sensor.observe(s, self.red_phi[i], self.rng) for i,s in enumerate(self.red)], [self.sensor.observe(s, self.blue_phi[i], self.rng) for i,s in enumerate(self.blue)])

    def _observations_both(self) -> tuple[np.ndarray, np.ndarray]:
        red_o, blue_o = self._observed(); ra, ba = [s.alive for s in self.red], [s.alive for s in self.blue]
        self.last_red_observations = build_team_observations(red_o, blue_o, ra, ba, **self.obs_cfg)
        self.last_blue_observations = build_team_observations(blue_o, red_o, ba, ra, **self.obs_cfg)
        return self.last_red_observations, self.last_blue_observations

    def _advance(self, states: list[AircraftState], phis: np.ndarray, actions: np.ndarray) -> None:
        for i,state in enumerate(states):
            if state.alive:
                _, control = self.controller.control_from_action(state, actions[i], self.spec)
                states[i] = self.integrator.step(state, control, self.dynamics, self.spec); phis[i] = control.phi

    def _record_death(self, team: str, index: int, cause: DeathCause) -> bool:
        states = self.red if team == "red" else self.blue; ledger = self.red_death_causes if team == "red" else self.blue_death_causes
        if ledger[index] != DeathCause.NONE or not states[index].alive: return False
        ledger[index] = cause; states[index].alive = False; return True

    def _resolve_boundaries(self) -> tuple[list[int], list[int]]:
        losses = ([], [])
        for ti,(team,states) in enumerate((("red",self.red),("blue",self.blue))):
            for i,s in enumerate(states):
                if s.alive and np.hypot(s.x,s.y) > self.radius and self._record_death(team,i,DeathCause.BOUNDARY): losses[ti].append(i)
        return losses

    @staticmethod
    def _snapshot(states: list[AircraftState]) -> list[AircraftState]:
        return [s.copy() for s in states]

    def _pre_attack_geometry(self, red: list[AircraftState], blue: list[AircraftState]) -> tuple[np.ndarray,list[int|None],list[int|None]]:
        geometric = np.zeros(4,np.float32); reward_targets=[]; threat_sources=[]
        for i,r in enumerate(red):
            target_i = self.fixed_policy.nearest_target_index(r,blue) if r.alive else None; reward_targets.append(target_i)
            red_g = compute_paper_geometry(r,blue[target_i]) if target_i is not None else None
            threats=[(j,compute_paper_geometry(b,r)) for j,b in enumerate(blue) if b.alive and r.alive]
            threat = min(threats,key=lambda x:(x[1].distance,x[0])) if threats else None; threat_sources.append(threat[0] if threat else None)
            geometric[i] = equation25_geometric_reward(red_g, threat[1] if threat else None)
        return geometric,reward_targets,threat_sources

    def _hit_proposals(self, red: list[AircraftState], blue: list[AircraftState]) -> tuple[list[tuple[int,int]],list[tuple[int,int]]]:
        proposals=([],[])
        for ti,(attackers,targets) in enumerate(((red,blue),(blue,red))):
            for i,attacker in enumerate(attackers):
                target_i=self.fixed_policy.nearest_target_index(attacker,targets) if attacker.alive else None
                if target_i is not None:
                    geometry=compute_paper_geometry(attacker,targets[target_i])
                    if self.weapon.can_fire(geometry) and self.weapon.sample_hit(geometry,self.rng): proposals[ti].append((i,target_i))
        return proposals

    def _apply_simultaneous_hits(self, red_hits: list[tuple[int,int]], blue_hits: list[tuple[int,int]]) -> tuple[list[tuple[int,int]],list[tuple[int,int]]]:
        """Every pre-attack successful proposal remains effective, including mutual hits."""
        credited=([],[])
        for ti,(team,hits) in enumerate((("blue",red_hits),("red",blue_hits))):
            by_target: dict[int,list[int]]={}
            for attacker,target in hits: by_target.setdefault(target,[]).append(attacker)
            for target,attackers in sorted(by_target.items()):
                if self._record_death(team,target,DeathCause.ATTACK): credited[ti].append((min(attackers),target))
        self.red_attack_kills += len(credited[0]); self.blue_attack_kills += len(credited[1])
        return credited

    def _outcome(self, terminated: bool, truncated: bool) -> str:
        red_alive,blue_alive=int(self.red_alive_mask.sum()),int(self.blue_alive_mask.sum())
        if red_alive==0 and blue_alive==0: return "draw_mutual_elimination"
        if blue_alive==0 and red_alive>0: return "red_win"
        if red_alive==0 and blue_alive>0: return "blue_win"
        if truncated: return "timeout"
        return "ongoing"

    def _summary(self, outcome: str, local: np.ndarray, red_actions: np.ndarray, blue_actions: np.ndarray, reward_targets, threat_sources) -> dict[str,Any]:
        rs,bs=death_summary(self.red_death_causes),death_summary(self.blue_death_causes)
        if sum(rs.values())!=4 or sum(bs.values())!=4: raise RuntimeError("death ledger conservation violated")
        return {"win":outcome=="red_win","red_win":outcome=="red_win","blue_win":outcome=="blue_win","draw_or_timeout":outcome.startswith("draw") or outcome=="timeout","environment_outcome":outcome,
            "attack_kills":self.red_attack_kills,"red_attack_kills":self.red_attack_kills,"blue_attack_kills":self.blue_attack_kills,"red_boundary_losses":rs["boundary_deaths"],"blue_boundary_losses":bs["boundary_deaths"],"boundary_losses":rs["boundary_deaths"]+bs["boundary_deaths"],
            "red_attack_deaths":rs["attack_deaths"],"blue_attack_deaths":bs["attack_deaths"],"red_survivors":rs["survivors"],"blue_survivors":bs["survivors"],"episode_length":self.steps,"local_rewards":local,
            "red_death_causes":self.red_death_causes.copy(),"blue_death_causes":self.blue_death_causes.copy(),"red_alive_mask":self.red_alive_mask,"blue_alive_mask":self.blue_alive_mask,
            "executed_red_actions":red_actions,"executed_blue_actions":blue_actions,"reward_target_indices":reward_targets,"threat_source_indices":threat_sources,"red_observations":self.last_red_observations,"blue_observations":self.last_blue_observations}

    def step(self, red_actions: np.ndarray, blue_actions: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray,bool,bool,dict[str,Any]]:
        red_actions=np.asarray(red_actions,np.float32)
        if red_actions.shape!=(4,3) or not np.all(np.isfinite(red_actions)): raise ValueError("red_actions must be finite with shape (4, 3)")
        if blue_actions is None: blue_actions=np.stack([self.fixed_policy.action(s,self.red)[0] for s in self.blue])
        else: blue_actions=np.asarray(blue_actions,np.float32)
        if blue_actions.shape!=(4,3) or not np.all(np.isfinite(blue_actions)): raise ValueError("blue_actions must be finite with shape (4, 3)")
        executed_red=np.clip(red_actions,-1,1)*self.red_alive_mask[:,None]; executed_blue=np.clip(blue_actions,-1,1)*self.blue_alive_mask[:,None]
        self._advance(self.red,self.red_phi,executed_red); self._advance(self.blue,self.blue_phi,executed_blue)
        red_boundary,blue_boundary=self._resolve_boundaries()
        pre_red,pre_blue=self._snapshot(self.red),self._snapshot(self.blue)
        geometric,reward_targets,threat_sources=self._pre_attack_geometry(pre_red,pre_blue)
        proposed_red,proposed_blue=self._hit_proposals(pre_red,pre_blue)
        red_hits,blue_hits=self._apply_simultaneous_hits(proposed_red,proposed_blue)
        local=geometric.copy()
        for i in range(4): local[i]+=equation25_event_reward(sum(a==i for a,_ in red_hits), i in [t for _,t in blue_hits], i in red_boundary)
        rewards=np.full(4,float(local.sum()),np.float32); self.steps+=1
        terminated=not any(s.alive for s in self.red) or not any(s.alive for s in self.blue); truncated=self.steps>=self.max_steps and not terminated
        outcome=self._outcome(terminated,truncated); red_obs,blue_obs=self._observations_both()
        info=self._summary(outcome,local,executed_red,executed_blue,reward_targets,threat_sources)
        return red_obs,rewards,terminated,truncated,info
