"""Small data structures and order-independent multi-aircraft damage resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import defaultdict
from typing import Sequence

import numpy as np

from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.combat.damage import DamageConfig, damage_for_random_value
from uav_env.core.state import UAVState
from uav_env.entities.uav import UAV


@dataclass(frozen=True)
class TargetAssignment:
    """Stable source-to-target maneuver assignment."""

    attacker_id: str
    target_id: str
    distance: float


@dataclass(frozen=True)
class AttackAttempt:
    """One sampled automatic attack before target aggregation."""

    attacker_id: str
    target_id: str
    distance: float
    random_value: float
    nominal_damage: float


@dataclass(frozen=True)
class ResolvedAttack:
    """One attack's allocated effective damage and destroy credit."""

    attacker_id: str
    target_id: str
    distance: float
    random_value: float
    nominal_damage: float
    effective_damage: float
    overkill_damage: float
    hit: bool
    destroy_credit: bool


@dataclass(frozen=True)
class MultiCombatStepResult:
    """Simultaneously updated states and complete attack records."""

    updated_states: dict[str, UAVState]
    attack_attempts: list[AttackAttempt]
    resolved_attacks: list[ResolvedAttack]


def assign_targets(attackers: Sequence[UAV], targets: Sequence[UAV]) -> list[TargetAssignment]:
    """Assign nearest distinct living targets before allowing target reuse."""

    living_attackers = sorted((u for u in attackers if u.is_alive), key=lambda u: u.uav_id)
    living_targets = sorted((u for u in targets if u.is_alive), key=lambda u: u.uav_id)
    unassigned = {u.uav_id for u in living_targets}
    assignments: list[TargetAssignment] = []
    for attacker in living_attackers:
        pool = [u for u in living_targets if u.uav_id in unassigned] or living_targets
        if not pool:
            break
        ranked = sorted(pool, key=lambda u: (float(np.linalg.norm(u.state.position_vector() - attacker.state.position_vector())), u.uav_id))
        target = ranked[0]
        distance = float(np.linalg.norm(target.state.position_vector() - attacker.state.position_vector()))
        assignments.append(TargetAssignment(attacker.uav_id, target.uav_id, distance))
        unassigned.discard(target.uav_id)
    return assignments


def resolve_multi_attacks(
    aircraft: Sequence[UAV],
    attack_config: AttackZoneConfig,
    damage_config: DamageConfig,
    rng: np.random.Generator,
    sample_team_order: tuple[int, ...] | None = None,
) -> MultiCombatStepResult:
    """Select nearest attackable targets, sample first, then update every target simultaneously."""

    ordered = sorted((u for u in aircraft if u.is_alive), key=lambda u: u.uav_id)
    if sample_team_order is not None:
        rank = {team: index for index, team in enumerate(sample_team_order)}
        sampling_order = sorted(ordered, key=lambda u: (rank.get(u.team, len(rank)), u.uav_id))
    else:
        sampling_order = ordered
    attempts: list[AttackAttempt] = []
    for attacker in sampling_order:
        candidates: list[tuple[float, str, UAV]] = []
        for target in ordered:
            if target.team == attacker.team:
                continue
            geometry = compute_combat_geometry(attacker.state, target.state, attack_config)
            if geometry.can_attack:
                candidates.append((geometry.distance, target.uav_id, target))
        if candidates:
            distance, _, target = min(candidates, key=lambda item: (item[0], item[1]))
            random_value = float(rng.random())
            attempts.append(AttackAttempt(attacker.uav_id, target.uav_id, distance, random_value, damage_for_random_value(random_value, damage_config)))

    by_target: dict[str, list[AttackAttempt]] = defaultdict(list)
    for attempt in attempts:
        by_target[attempt.target_id].append(attempt)
    states = {u.uav_id: u.state.copy() for u in aircraft}
    resolved: list[ResolvedAttack] = []
    for target_id in sorted(by_target):
        target_state = states[target_id]
        target_attempts = by_target[target_id]
        total_nominal = sum(a.nominal_damage for a in target_attempts)
        total_effective = min(target_state.health, total_nominal)
        allocations = {
            a.attacker_id: (total_effective * a.nominal_damage / total_nominal if total_nominal > 0.0 else 0.0)
            for a in target_attempts
        }
        destroyed = total_effective > 0.0 and target_state.health - total_effective <= 0.0
        credit_id: str | None = None
        if destroyed:
            credit_id = min(target_attempts, key=lambda a: (-allocations[a.attacker_id], a.distance, a.attacker_id)).attacker_id
        for attempt in sorted(target_attempts, key=lambda a: a.attacker_id):
            effective = allocations[attempt.attacker_id]
            resolved.append(ResolvedAttack(
                attempt.attacker_id, target_id, attempt.distance, attempt.random_value, attempt.nominal_damage,
                effective, attempt.nominal_damage - effective, effective > 0.0, attempt.attacker_id == credit_id,
            ))
        health_after = max(0.0, target_state.health - total_effective)
        states[target_id] = replace(
            target_state,
            health=health_after,
            ever_hit=target_state.ever_hit or total_effective > 0.0,
            alive=target_state.alive and not destroyed,
            damaged=target_state.damaged or destroyed,
        )
    return MultiCombatStepResult(states, attempts, resolved)
