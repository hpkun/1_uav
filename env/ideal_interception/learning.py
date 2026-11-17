"""PG-MAPPO observation and reward adapter around the unchanged V3.5 environment."""
from copy import deepcopy
import math

import numpy as np

from prediction.history import IntruderHistory
from prediction.intercept_prediction import estimate_three_point_pip_from_history

from .config import DEFENDER_SLOTS, INTRUDER_SLOTS, PROTOCOL
from .environment import InterceptionEnv
from .geometry import closest

PG_OBS_DIM = 6 + 8 + 5 + 4*7 + 1
SELF = slice(0, 6)
TARGET = slice(6, 14)
PIP = slice(14, 19)
FRIENDS = slice(19, 47)
REMAINING = 47
FRIEND_COUNT = 4
LEARNING_PROTOCOL = "prediction-guided-mappo-v1"


def usable_pip(prediction, time, config):
    """Restrict a solver result to this episode and arena, without changing it."""
    tau = prediction.horizon_s
    point = prediction.predicted_target_position_m
    if (prediction.status != "READY" or tau is None or not math.isfinite(tau)
            or tau <= 0 or tau > config.dt*config.max_steps-time or point is None):
        return False
    point = np.asarray(point, dtype=float)
    return bool(point.shape == (3,) and np.isfinite(point).all()
                and np.all(point >= config.lower) and np.all(point <= config.upper))


def build_pg_observation(info, history, config):
    """Pure view of current state and already observed position history."""
    defenders = np.asarray(info["defenders"], dtype=float)
    intruders = np.asarray(info["intruders"], dtype=float)
    active = np.asarray(info["defender_mask"], dtype=bool)
    targets_active = np.asarray(info["intruder_mask"], dtype=bool)
    if (defenders.shape != (DEFENDER_SLOTS, 6) or
            intruders.shape != (INTRUDER_SLOTS, 6) or
            active.shape != (DEFENDER_SLOTS,) or
            targets_active.shape != (INTRUDER_SLOTS,)):
        raise ValueError("invalid PG state shape")
    output = np.zeros((DEFENDER_SLOTS, PG_OBS_DIM), dtype=np.float32)
    priors = {}
    center = np.asarray(config.protected_center, dtype=float)
    total = config.dt*config.max_steps
    remaining = max(0., total-float(info["time"]))
    for i in np.flatnonzero(active):
        i = int(i)
        if i not in info["assignment"]:
            raise ValueError(f"active fixed-assignment defender D{i} has no target")
        j = int(info["assignment"][i])
        if not 0 <= j < INTRUDER_SLOTS or not targets_active[j]:
            raise ValueError(f"D{i} has inactive or invalid assigned target")
        own = defenders[i]
        target_position = intruders[j, :3]  # Never read intruder velocity slots.
        relative = target_position-own[:3]
        distance = float(np.linalg.norm(relative))
        target_direction = relative/distance if distance > 1e-12 else np.zeros(3)
        prediction = estimate_three_point_pip_from_history(
            history, j, own[:3], config.defender_speeds[i])
        velocity = prediction.estimated_target_velocity_mps
        velocity_valid = bool(velocity is not None and np.shape(velocity) == (3,)
                              and np.isfinite(velocity).all())
        estimated = np.asarray(velocity, dtype=float) if velocity_valid else np.zeros(3)
        valid = usable_pip(prediction, float(info["time"]), config)
        point = np.asarray(prediction.predicted_target_position_m, dtype=float) if valid else None
        pip_block = np.zeros(5)
        if valid:
            delta = point-own[:3]
            pip_distance = float(np.linalg.norm(delta))
            if not math.isfinite(pip_distance) or pip_distance <= 1e-12:
                valid, point = False, None
            else:
                # V2: unit line-of-sight direction; horizon carries the time/range scale.
                pip_block = np.r_[delta/pip_distance, prediction.horizon_s/total, 1.]
        peers = sorted((float(np.linalg.norm(defenders[k, :3]-own[:3])), int(k))
                       for k in np.flatnonzero(active) if k != i)[:FRIEND_COUNT]
        neighbor_blocks = [np.r_[(defenders[k, :3]-own[:3])/config.position_scale,
                                 (defenders[k, 3:]-own[3:])/config.speed_scale, 1.]
                           for _, k in peers]
        neighbor_blocks += [np.zeros(7) for _ in range(FRIEND_COUNT-len(peers))]
        output[i] = np.concatenate(((own[:3]-center)/config.position_scale,
            own[3:]/config.speed_scale, target_direction,
            [distance/config.position_scale], estimated/config.speed_scale,
            [float(velocity_valid)], pip_block, *neighbor_blocks, [remaining/total]))
        priors[i] = dict(target=j, target_position=target_position.copy(),
                         pip_valid=valid, pip_point=point.copy() if valid else None,
                         prediction=prediction)
    if output.shape != (DEFENDER_SLOTS, PG_OBS_DIM) or not np.isfinite(output).all():
        raise FloatingPointError("nonfinite PG observation")
    return output, priors


def guidance_reward(before, after, frozen_point, speed, dt):
    start = np.asarray(before, dtype=float)
    end = np.asarray(after, dtype=float)
    point = np.asarray(frozen_point, dtype=float)
    if any(value.shape != (3,) or not np.isfinite(value).all()
           for value in (start, end, point)) or speed <= 0 or dt <= 0:
        raise ValueError("invalid guidance reward inputs")
    return .01*(np.linalg.norm(point-start)-np.linalg.norm(point-end))/(speed*dt)


def safety_reward(distance):
    if distance is None or distance >= 8.:
        return 0.
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("invalid friendly distance")
    if distance >= 3.:
        return -.002*((8.-distance)/5.)**2
    return -.002-.018*((3.-distance)/3.)**2


def minimum_friendly_distances(before, after):
    """Continuous step-internal pair minima, ending at each member's exit."""
    active = np.flatnonzero(before["defender_mask"])
    starts = np.asarray(before["defenders"], dtype=float)
    velocities = np.asarray(after["executed_velocities"], dtype=float)
    exits = {event["members"][0]:float(event["time"])
             for event in after["events"] if event["kind"] == "member_inactive"
             and event["members"][0] < DEFENDER_SLOTS}
    t0, t1 = float(before["time"]), float(after["time"])
    result = {int(i):None for i in active}
    for offset, i in enumerate(active):
        for k in active[offset+1:]:
            duration = max(0., min(exits.get(int(i), t1), exits.get(int(k), t1))-t0)
            distance = closest(starts[i, :3], velocities[i],
                               starts[k, :3], velocities[k], duration)
            for member in (int(i), int(k)):
                current = result[member]
                result[member] = distance if current is None else min(current, distance)
    return result


def event_rewards(events, original_assignment, start_mask):
    rewards = np.zeros(DEFENDER_SLOTS, dtype=np.float64)
    owners = {int(j):int(i) for i, j in original_assignment.items()}
    for event in events:
        kind, members = event["kind"], event["members"]
        if kind == "interception_contact":
            i, j = members[0], members[1]-DEFENDER_SLOTS
            if start_mask[i]:
                rewards[i] += 10. if original_assignment[i] == j else -15.
        elif kind == "region_breach":
            j = members[0]-DEFENDER_SLOTS
            i = owners.get(j)
            if i is not None and start_mask[i]:
                rewards[i] -= 12.
        elif kind == "friendly_collision":
            for i in members:
                if start_mask[i]:
                    rewards[i] -= 15.
        elif kind == "boundary_exit" and members[0] < DEFENDER_SLOTS:
            if start_mask[members[0]]:
                rewards[members[0]] -= 15.
    return rewards


class PredictionGuidedInterceptionEnv:
    def __init__(self, config):
        if config.scale not in ("6v6", "15v15") or config.assignment_policy != "hungarian_fixed":
            raise ValueError("PG learning requires fixed Hungarian 6v6 or 15v15")
        self.config = config
        self.base = InterceptionEnv(config)
        self.history = IntruderHistory(config.dt, slots=INTRUDER_SLOTS)

    def reset(self, seed=None):
        _, info = self.base.reset(seed)
        self.history.reset(info["scenario_id"])
        self.history.observe(info["time"], info["intruders"][:, :3],
                             info["intruder_mask"], info["scenario_id"])
        observation, _ = build_pg_observation(info, self.history, self.config)
        return observation, info

    def step(self, actions):
        before = self.base.output()[1]
        _, priors = build_pg_observation(before, self.history, self.config)
        start_mask = before["defender_mask"].copy()
        _, _, terminated, truncated, after = self.base.step(actions)
        distances = minimum_friendly_distances(before, after)
        events = event_rewards(after["events"], before["original_assignment"], start_mask)
        rewards = np.zeros(DEFENDER_SLOTS, dtype=np.float32)
        parts = dict(guide=0., safe=0., event=0., time=0.)
        for i in np.flatnonzero(start_mask):
            prior = priors[int(i)]
            point = prior["pip_point"] if prior["pip_valid"] else prior["target_position"]
            guide = guidance_reward(before["defenders"][i, :3],
                after["defenders"][i, :3], point, self.config.defender_speeds[i], self.config.dt)
            safe = safety_reward(distances[int(i)])
            event = events[i]
            time_cost = -.0003
            rewards[i] = guide+safe+event+time_cost
            for key, value in (("guide", guide), ("safe", safe),
                               ("event", event), ("time", time_cost)):
                parts[key] += float(value)
        if rewards.shape != (DEFENDER_SLOTS,) or not np.isfinite(rewards).all():
            raise FloatingPointError("nonfinite PG learning reward")
        after["learning_reward_sum"] = float(rewards.sum(dtype=np.float64))
        after["learning_reward_parts"] = parts
        self.history.observe(after["time"], after["intruders"][:, :3],
                             after["intruder_mask"], after["scenario_id"])
        observation, _ = build_pg_observation(after, self.history, self.config)
        return observation, rewards, terminated, truncated, after

    def snapshot(self):
        return dict(protocol=LEARNING_PROTOCOL, environment_protocol=PROTOCOL,
                    base=self.base.snapshot(), history=deepcopy(self.history))

    def restore(self, snapshot):
        if (snapshot.get("protocol") != LEARNING_PROTOCOL or
                snapshot.get("environment_protocol") != PROTOCOL):
            raise ValueError("incompatible PG learning snapshot")
        self.base.restore(snapshot["base"])
        history = deepcopy(snapshot["history"])
        if not isinstance(history, IntruderHistory) or history.dt != self.config.dt or history.slots != INTRUDER_SLOTS:
            raise ValueError("incompatible PG history snapshot")
        self.history = history
