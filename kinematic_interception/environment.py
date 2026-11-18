"""Independent reference-target simulation with curved-defender substep events."""
from dataclasses import dataclass
import math

import numpy as np

from env.ideal_interception.assignment import nearest_distance_assignment
from env.ideal_interception.geometry import boundary, closest, proximity, region_entry
from env.ideal_interception.large_scale_20v20_env import TwentyConfig, TwentyVTwentyEnv
from env.ideal_interception.trajectory import Reference

from . import PROTOCOL
from .dynamics import Command, DynamicsConfig, PointState, WingState, step, velocity


PRIORITY = {"region_breach": 0, "friendly_collision": 1,
            "interception_contact": 2, "intruder_collision": 3,
            "boundary_exit": 4, "path_complete": 5}


@dataclass(frozen=True)
class Scene:
    defender_starts: np.ndarray
    paths: tuple[Reference, ...]
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    protected_center: tuple[float, float, float]
    protected_radius: float
    protected_height: float
    contact_distance: float = .1
    max_steps: int = 600
    identity: str = "synthetic-1v1"

    def __post_init__(self):
        starts = np.asarray(self.defender_starts, float)
        lower = np.asarray(self.lower,float)
        upper = np.asarray(self.upper,float)
        center = np.asarray(self.protected_center,float)
        if (starts.ndim != 2 or starts.shape[1] != 3 or not len(starts)
                or not self.paths or not np.isfinite(starts).all()
                or lower.shape != (3,) or upper.shape != (3,) or center.shape != (3,)
                or not np.isfinite(np.r_[lower,upper,center]).all()
                or np.any(lower >= upper) or self.contact_distance <= 0
                or not math.isfinite(self.contact_distance)
                or not math.isfinite(self.protected_radius)
                or not math.isfinite(self.protected_height)
                or self.protected_radius <= 0 or self.protected_height <= 0
                or self.max_steps < 1):
            raise ValueError("invalid independent scene")
        initial = np.array([path.sample(0.)[0] for path in self.paths])
        points = np.vstack((starts, initial))
        if np.any(points <= self.lower) or np.any(points >= self.upper):
            raise ValueError("scene member outside arena")
        if any(np.linalg.norm(points[i]-points[j]) <= self.contact_distance
               for i in range(len(points)) for j in range(i+1, len(points))):
            raise ValueError("initial geometric contact")
        if any(np.linalg.norm(point[:2]-center[:2]) <= self.protected_radius
               and center[2] <= point[2] <= center[2]+self.protected_height
               for point in initial):
            raise ValueError("Intruder initially inside protected region")


def one_vs_one_scene(seed=17, curved=True):
    """Fixed geometry plus seeded lateral offset; identical across A/B and G0-G3."""
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(-6., 6.))
    start = np.array((65., 12.+offset, 30.))
    end = np.array((-145., 0., 30.))
    controls = (np.array((10., 34.+offset, 32.)),
                np.array((-80., -20.+offset, 28.)))
    path = Reference(start, end, 20., "bezier" if curved else "straight",
                     control_points=controls if curved else None)
    return Scene(np.array(((-100., -25., 30.),)), (path,),
                 (-180., -100., 0.), (180., 100., 100.),
                 (-140., 0., 0.), 4., 100., identity=f"1v1-{seed}-{'curve' if curved else 'line'}")


def twenty_scene(seed=17, count=20):
    """Read the historical scene generator once; never call its step or alter it."""
    if count not in (3, 6, 20):
        raise ValueError("multi-member scene supports 3, 6 or 20 pairs")
    old = TwentyVTwentyEnv(TwentyConfig(seed=seed))
    old.reset(seed)
    c = old.config
    return Scene(old.defenders[:count, :3].copy(), tuple(old.paths[:count]),
                 c.lower, c.upper, c.protected_center, c.protected_radius,
                 c.protected_height, c.contact_distance, c.max_steps,
                 identity=f"{count}v{count}-{old.scenario_id}")


class KinematicInterceptionEnv:
    def __init__(self, scene: Scene, model: str, dynamics=None, substep=.01):
        if model not in ("A", "B") or substep <= 0 or substep > .01:
            raise ValueError("model A/B and substep no larger than .01 s required")
        self.scene, self.model = scene, model
        self.dynamics = dynamics or DynamicsConfig()
        self.substep = substep
        self.reset()

    def reset(self):
        self.states = [PointState(p.copy(), np.array((20., 0., 0.))) if self.model == "A"
                       else WingState(p.copy(), 20., 0., 0.)
                       for p in np.asarray(self.scene.defender_starts)]
        self.defender_mask = np.ones(len(self.states), bool)
        self.intruder_mask = np.ones(len(self.scene.paths), bool)
        self.defender_status = ["active"]*len(self.states)
        self.intruder_status = ["active"]*len(self.scene.paths)
        self.intruder_exit_positions = [None]*len(self.scene.paths)
        self.time = 0.
        self.assignment = nearest_distance_assignment(self.defender_positions(),
            self.intruder_positions(), self.defender_mask, self.intruder_mask)
        self.step_count = 0
        self.outcome = "ongoing"
        self.events = []
        self.interception_count = self.breach_count = 0
        self.unassigned_contact_count = self.friendly_collision_count = 0
        self.intruder_collision_count = self.defender_boundary_count = 0
        self.minimum_separation = None
        self.trajectory_length = np.zeros(len(self.states))
        self.max_applied_acceleration = 0.
        self.max_yaw_rate = 0.
        self.max_flightpath_rate = 0.
        self.max_lateral_acceleration = 0.
        self.min_active_speed = 20.
        return self.snapshot()

    def defender_positions(self):
        return np.array([state.position for state in self.states])

    def defender_velocities(self):
        return np.array([velocity(state) if self.defender_mask[i] else np.zeros(3)
                         for i, state in enumerate(self.states)])

    def intruder_positions(self):
        return np.array([path.sample(self.time)[0] if self.intruder_mask[j]
                         else self.intruder_exit_positions[j]
                         for j,path in enumerate(self.scene.paths)])

    def snapshot(self):
        kinematic = np.array([np.r_[state.position,
            state.velocity if isinstance(state,PointState) else
            np.array((state.speed,state.yaw,state.flightpath))]
            for state in self.states])
        return dict(protocol=PROTOCOL, model=self.model, scene_id=self.scene.identity,
                    time=self.time, step=self.step_count, outcome=self.outcome,
                    defenders=np.column_stack((self.defender_positions(),
                                               self.defender_velocities())),
                    defender_kinematic_states=kinematic,
                    intruders=np.array([np.r_[self.intruder_positions()[j],
                        path.sample(self.time)[1] if self.intruder_mask[j] else np.zeros(3)]
                        for j, path in enumerate(self.scene.paths)]),
                    defender_mask=self.defender_mask.copy(),
                    intruder_mask=self.intruder_mask.copy(), assignment=self.assignment.copy(),
                    defender_status=self.defender_status.copy(),
                    intruder_status=self.intruder_status.copy(),
                    intercepted_count=self.interception_count,
                    breached_count=self.breach_count,
                    unassigned_contact_count=self.unassigned_contact_count,
                    friendly_collision_count=self.friendly_collision_count,
                    minimum_defender_separation_m=self.minimum_separation)

    def _positions_at(self, starts, ends, target_start, target_end, fraction):
        defenders = np.array([a.position+fraction*(b.position-a.position)
                              for a,b in zip(starts,ends)])
        intruders = target_start+fraction*(target_end-target_start)
        return defenders, intruders

    def _resolve(self, kind, members, at, position_arrays):
        dpos, ipos = position_arrays
        n = len(self.states)
        event_kind = kind
        if kind == "interception_contact" and self.assignment.get(members[0]) != members[1]-n:
            event_kind = "unassigned_contact"
        event = dict(kind=event_kind, time=float(at), members=list(members),
                     positions=[(dpos[m] if m < n else ipos[m-n]).tolist() for m in members])
        self.events.append(event)

        def retire(member, status):
            if member < n:
                self.defender_mask[member] = False
                self.defender_status[member] = status
                self.states[member].position = dpos[member].copy()
            else:
                self.intruder_mask[member-n] = False
                self.intruder_status[member-n] = status
                self.intruder_exit_positions[member-n] = ipos[member-n].copy()

        if kind == "interception_contact":
            if event_kind == kind:
                self.interception_count += 1
                retire(members[0], "contact_exit")
                retire(members[1], "intercepted")
            else:
                self.unassigned_contact_count += 1
                retire(members[0], "unassigned_contact_exit")
                retire(members[1], "unassigned_contact_exit")
        elif kind == "region_breach":
            self.breach_count += 1
            retire(members[0], "breached")
        elif kind == "friendly_collision":
            self.friendly_collision_count += 1
            for member in members:
                retire(member, "friendly_contact_exit")
        elif kind == "intruder_collision":
            self.intruder_collision_count += 1
            for member in members:
                retire(member, "intruder_contact_exit")
        else:
            for member in members:
                if kind == "boundary_exit" and member < n:
                    self.defender_boundary_count += 1
                retire(member, kind)
        self.assignment = nearest_distance_assignment(dpos, ipos,
            self.defender_mask, self.intruder_mask)
        if not self.intruder_mask.any():
            if self.interception_count == len(self.scene.paths):
                self.outcome = "all_intercepted"
            elif len(self.scene.paths) == 1:
                self.outcome = self.intruder_status[0]
            else:
                self.outcome = "mixed_resolution"

    def step(self, commands):
        if self.outcome != "ongoing":
            raise RuntimeError("reset required after episode resolution")
        if len(commands) != len(self.states):
            raise ValueError("one high-level command per Defender slot required")
        for i in np.flatnonzero(self.defender_mask):
            if not isinstance(commands[i], Command):
                raise TypeError("active Defender command must be Command")
            commands[i].validated(self.dynamics.v_max)
        origin = self.time
        stop = min(origin+self.dynamics.dt, self.scene.max_steps*self.dynamics.dt)
        while self.time < stop-1e-12 and self.outcome == "ongoing":
            right = min(stop, self.time+self.substep)
            cuts = {self.time, right}
            for j in np.flatnonzero(self.intruder_mask):
                for left, end, _, _ in self.scene.paths[j].segments(self.time, right):
                    cuts.update((left,end))
            ordered = sorted(cuts)
            for left,right in zip(ordered[:-1],ordered[1:]):
                if self.outcome != "ongoing":
                    break
                h = right-left
                if h <= 1e-12:
                    continue
                start_states = [state.copy() for state in self.states]
                end_states = []
                diagnostics = {}
                for i,state in enumerate(start_states):
                    if not self.defender_mask[i]:
                        end_states.append(state.copy())
                        continue
                    end,acceleration,diagnostic = step(state,commands[i],self.dynamics,h)
                    end_states.append(end)
                    diagnostics[i] = (acceleration,diagnostic)
                target_start = np.array([path.sample(left)[0] for path in self.scene.paths])
                target_end = np.array([path.sample(right)[0] for path in self.scene.paths])
                dv = np.array([(b.position-a.position)/h for a,b in zip(start_states,end_states)])
                iv = (target_end-target_start)/h
                candidates = []
                n = len(self.states)

                def add(kind, tau, members):
                    if tau is not None and -1e-9 <= tau <= h+1e-9:
                        candidates.append((float(np.clip(tau,0.,h)), PRIORITY[kind], members, kind))

                active_d = np.flatnonzero(self.defender_mask)
                active_i = np.flatnonzero(self.intruder_mask)
                exit_times = {}
                separation_cursor = 0.

                def update_separation(until):
                    nonlocal separation_cursor
                    duration = until-separation_cursor
                    if duration < -1e-10:
                        raise FloatingPointError("event order reversed")
                    ids = np.flatnonzero(self.defender_mask)
                    for a,i in enumerate(ids):
                        for j in ids[a+1:]:
                            value = closest(start_states[i].position+dv[i]*separation_cursor,
                                dv[i],start_states[j].position+dv[j]*separation_cursor,
                                dv[j],max(0.,duration))
                            self.minimum_separation = (value if self.minimum_separation is None
                                                       else min(self.minimum_separation,value))
                    separation_cursor = until

                for i in active_d:
                    add("boundary_exit", boundary(start_states[i].position,dv[i],
                        np.asarray(self.scene.lower),np.asarray(self.scene.upper),h+1e-9), (int(i),))
                for a,i in enumerate(active_d):
                    for j in active_d[a+1:]:
                        add("friendly_collision", proximity(start_states[i].position,dv[i],
                            start_states[j].position,dv[j],self.scene.contact_distance,h+1e-9),
                            (int(i),int(j)))
                for j in active_i:
                    j = int(j)
                    add("region_breach", region_entry(target_start[j],iv[j],
                        np.asarray(self.scene.protected_center),self.scene.protected_radius,
                        self.scene.protected_height,h+1e-9), (n+j,))
                    add("boundary_exit", boundary(target_start[j],iv[j],
                        np.asarray(self.scene.lower),np.asarray(self.scene.upper),h+1e-9), (n+j,))
                    if left < self.scene.paths[j].duration <= right+1e-10:
                        add("path_complete",self.scene.paths[j].duration-left,(n+j,))
                    for i in active_d:
                        add("interception_contact",proximity(start_states[i].position,dv[i],
                            target_start[j],iv[j],self.scene.contact_distance,h+1e-9),
                            (int(i),n+j))
                for a,j in enumerate(active_i):
                    for k in active_i[a+1:]:
                        add("intruder_collision",proximity(target_start[j],iv[j],
                            target_start[k],iv[k],self.scene.contact_distance,h+1e-9),
                            (n+int(j),n+int(k)))
                for tau, _, members, kind in sorted(candidates):
                    if self.outcome != "ongoing" or any(
                        not (self.defender_mask[m] if m<n else self.intruder_mask[m-n])
                        for m in members):
                        continue
                    positions = self._positions_at(start_states,end_states,target_start,
                                                   target_end,tau/h)
                    update_separation(tau)
                    for member in members:
                        if member < n and self.defender_mask[member]:
                            exit_times[member] = tau
                            self.trajectory_length[member] += float(np.linalg.norm(
                                positions[0][member]-start_states[member].position))
                            partial = (step(start_states[member],commands[member],
                                            self.dynamics,tau)[0] if tau > 0
                                       else start_states[member].copy())
                            partial.position = positions[0][member].copy()
                            self.states[member] = partial
                    self._resolve(kind,members,left+tau,positions)
                elapsed = (self.outcome != "ongoing")
                if not elapsed:
                    update_separation(h)
                for i in active_d:
                    duration = exit_times.get(i,separation_cursor)
                    if self.defender_mask[i]:
                        if elapsed:
                            partial = (step(start_states[i],commands[i],self.dynamics,
                                            duration)[0] if duration > 0
                                       else start_states[i].copy())
                            partial.position = start_states[i].position+dv[i]*duration
                            self.states[i] = partial
                        else:
                            self.states[i] = end_states[i]
                        self.trajectory_length[i] += float(np.linalg.norm(
                            self.states[i].position-start_states[i].position))
                    if duration > 0:
                        if i in exit_times or elapsed:
                            _,acceleration,diagnostic = step(start_states[i],commands[i],
                                                              self.dynamics,duration)
                        else:
                            acceleration,diagnostic = diagnostics[i]
                        self.max_applied_acceleration = max(self.max_applied_acceleration,
                            float(np.linalg.norm(acceleration)))
                        self.min_active_speed = min(self.min_active_speed,
                            float(np.linalg.norm(velocity(self.states[i]))))
                        if self.model == "B":
                            self.max_yaw_rate = max(self.max_yaw_rate,abs(diagnostic["yaw_rate"]))
                            self.max_flightpath_rate = max(self.max_flightpath_rate,
                                abs(diagnostic["flightpath_rate"]))
                            self.max_lateral_acceleration = max(self.max_lateral_acceleration,
                                diagnostic["lateral_acceleration"])
                self.time = (self.events[-1]["time"] if elapsed else right)
                if elapsed:
                    break
        self.step_count += 1
        if self.outcome == "ongoing" and self.step_count >= self.scene.max_steps:
            self.outcome = "deadline_failure"
        return self.snapshot()
