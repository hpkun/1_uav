"""Continuous-time playback of recorded ideal point-mass motion."""
from bisect import bisect_right
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from env.ideal_interception.config import PROTOCOL,SCALES,DEFENDER_SLOTS,INTRUDER_SLOTS
from env.ideal_interception.trajectory import Reference
from env.ideal_interception import InterceptionConfig
from env.ideal_interception.assignment import nearest_distance_assignment,hungarian_fixed_assignment
from .ideal_interception_trace import TRACE_VERSION

PHYSICAL_EVENTS = {"interception_contact","friendly_collision","intruder_collision",
                   "region_breach","boundary_exit","path_complete"}


class Timeline:
    def __init__(self,source):
        opener=gzip.open if Path(source).suffix==".gz" else open
        with opener(source,"rt",encoding="utf-8") as stream:
            data=json.load(stream)
        if (data.get("trace_version") not in (TRACE_VERSION,"uav-ideal-trace-v1") or
                data.get("environment_protocol") not in (PROTOCOL,"uav-ideal-interception-v3.3.0",
                                                         "uav-ideal-interception-v3.2.0",
                                                         "uav-ideal-interception-v3.1.0",
                                                         "uav-ideal-interception-v3.0.0",
                                                         "uav-ideal-interception-v2.0.0")):
            raise ValueError("unsupported ideal trajectory protocol")
        self.data=data
        legacy_slots=data["environment_protocol"] not in (PROTOCOL,"uav-ideal-interception-v3.3.0")
        self.defender_slots=5 if legacy_slots else DEFENDER_SLOTS
        self.intruder_slots=2 if legacy_slots else INTRUDER_SLOTS
        try:
            raw=data["config"]
            if legacy_slots:
                d,i=SCALES[raw["scale"]]
                if (len(raw["defender_speeds"])!=5 or len(raw["intruder_speeds"])!=2
                        or len(raw["defender_starts"])!=5 or len(raw["intruder_starts"])!=2):
                    raise ValueError("invalid historical 5/2 slot schema")
                self.config=SimpleNamespace(**raw,defender_count=d,intruder_count=i)
            else:
                self.config=InterceptionConfig(**raw)
            self.frames=data["frames"]
            references=data["references"]
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f"invalid ideal trace configuration or required field: {exc}") from exc
        self.times=[float(frame["time"]) for frame in self.frames]
        if len(self.times)<2 or self.times[0]!=0 or any(not np.isfinite(t) for t in self.times):
            raise ValueError("invalid ideal trajectory clock")
        if any(b<=a for a,b in zip(self.times,self.times[1:])):
            raise ValueError("ideal trajectory times must increase")
        self.duration=self.times[-1]
        self.events=[event for frame in self.frames[1:] for event in frame["events"]]
        if any(event["time"]<0 or event["time"]>self.duration for event in self.events):
            raise ValueError("event outside recorded episode")
        self.key_times=sorted(set(self.times+[event["time"] for event in self.events]))
        self.references=[]
        if len(references)!=self.config.intruder_count:
            raise ValueError("reference/entity count mismatch")
        for j,reference in enumerate(references):
            points=np.asarray(reference["points"],float)
            spec=reference["specification"]
            if (points.ndim!=2 or points.shape[1]!=3 or len(points)!=spec["polyline_points"]
                    or not np.isfinite(points).all() or
                    hashlib.sha256(points.tobytes()).hexdigest()!=spec["polyline_sha256"]):
                raise ValueError("recorded ideal reference hash mismatch")
            expected=Reference(spec["start"],spec["end"],spec["speed"],spec["mode"],
                               control_points=spec["controls"] if spec["mode"]=="bezier" else None)
            if (spec != expected.specification() or
                    not np.array_equal(points,expected.points) or
                    spec["speed"]!=self.config.intruder_speeds[j]):
                raise ValueError(f"reference {j} specification, speed or duration mismatch")
            distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            if np.any(np.diff(distances)<=0):
                raise ValueError(f"reference {j} has zero-length segment")
            self.references.append((points,distances,spec))
        if len(self.references)!=sum(self.frames[0]["intruder_mask"]):
            raise ValueError("reference/entity count mismatch")
        self.validate_trace()

    def validate_trace(self):
        """Cross-check each recorded state against executed motion and event transitions."""
        tolerance=dict(rtol=1e-10,atol=1e-8)
        initial=self.frames[0]
        if (self.data["environment_protocol"] in (PROTOCOL,"uav-ideal-interception-v3.3.0") and
                self.config.assignment_policy=="hungarian_fixed"):
            original={int(i):int(j) for i,j in initial["assignment"].items()}
            if self.data.get("original_assignment")!={str(i):j for i,j in original.items()}:
                raise ValueError("fixed assignment metadata mismatch")
            distances={str(i):float(np.linalg.norm(
                np.asarray(initial["defenders"][i][:3])-
                np.asarray(initial["intruders"][j][:3]))) for i,j in original.items()}
            if (self.data.get("per_pair_initial_distance_m")!=distances or
                    not np.isclose(self.data.get("assignment_total_cost_m",np.nan),
                                   sum(distances.values()),**tolerance)):
                raise ValueError("fixed assignment cost metadata mismatch")
        for k,frame in enumerate(self.frames):
            for field,shape in (("defenders",(self.defender_slots,6)),
                                ("intruders",(self.intruder_slots,6))):
                if np.shape(frame.get(field))!=shape or not np.isfinite(np.asarray(frame[field],float)).all():
                    raise ValueError(f"frame {k} invalid {field}")
            for field,size in (("defender_mask",self.defender_slots),
                               ("intruder_mask",self.intruder_slots)):
                mask=frame.get(field)
                if not isinstance(mask,list) or len(mask)!=size or any(type(v) is not bool for v in mask):
                    raise ValueError(f"frame {k} invalid {field}")
            if type(frame.get("events")) is not list:
                raise ValueError(f"frame {k} invalid events")
            if self.data["trace_version"] == TRACE_VERSION:
                assignment=frame.get("assignment")
                if not isinstance(assignment,dict):
                    raise ValueError(f"frame {k} missing recorded assignment")
                try:
                    pairs={int(i):int(j) for i,j in assignment.items()}
                except (TypeError,ValueError) as exc:
                    raise ValueError(f"frame {k} invalid assignment") from exc
                if (len(pairs)!=len(assignment) or any(str(i) != str(key) for key,i in
                        ((key,int(key)) for key in assignment)) or
                        any(not frame["defender_mask"][i] or not frame["intruder_mask"][j]
                            for i,j in pairs.items() if 0<=i<self.defender_slots and 0<=j<self.intruder_slots) or
                        any(i not in range(self.defender_slots) or j not in range(self.intruder_slots)
                            for i,j in pairs.items())):
                    raise ValueError(f"frame {k} invalid assignment")
                if (frame["outcome"]=="ongoing" and any(frame["intruder_mask"]) and
                        set(pairs)!={i for i,active in enumerate(frame["defender_mask"]) if active}):
                    raise ValueError(f"frame {k} missing active defender assignment")
                if self.data["environment_protocol"] in (PROTOCOL,"uav-ideal-interception-v3.3.0") and self.config.assignment_policy=="hungarian_fixed":
                    original={int(i):int(j) for i,j in self.frames[0]["assignment"].items()}
                    expected=hungarian_fixed_assignment(
                        np.asarray(initial["defenders"])[:self.config.defender_count,:3],
                        np.asarray(initial["intruders"])[:self.config.intruder_count,:3])
                    if original!=expected or pairs!={i:j for i,j in original.items()
                            if frame["defender_mask"][i] and frame["intruder_mask"][j]}:
                        raise ValueError(f"frame {k} fixed assignment mismatch")
                elif self.data["environment_protocol"] in (PROTOCOL,"uav-ideal-interception-v3.3.0","uav-ideal-interception-v3.2.0",
                    "uav-ideal-interception-v3.1.0",
                    "uav-ideal-interception-v3.0.0") and frame["outcome"]=="ongoing":
                    expected=nearest_distance_assignment(np.asarray(frame["defenders"])[:,:3],
                        np.asarray(frame["intruders"])[:,:3],frame["defender_mask"],frame["intruder_mask"])
                    if k==0 or (self.frames[k-1]["defender_mask"]!=frame["defender_mask"] or
                                 self.frames[k-1]["intruder_mask"]!=frame["intruder_mask"]):
                        if pairs!=expected:
                            raise ValueError(f"frame {k} recorded assignment mismatch")
                    elif pairs!={int(i):int(j) for i,j in self.frames[k-1]["assignment"].items()}:
                        raise ValueError(f"frame {k} assignment changed without member event")
        if initial["events"] or initial["step"]!=0:
            raise ValueError("frame 0 invalid initial state")
        combined=lambda frame: np.r_[np.asarray(frame["defenders"],float),np.asarray(frame["intruders"],float)]
        masks=lambda frame: list(frame["defender_mask"])+list(frame["intruder_mask"])
        all_events=[]
        for k in range(1,len(self.frames)):
            before,after=self.frames[k-1],self.frames[k]
            t0,t1=self.times[k-1],self.times[k]
            control=after.get("control")
            if control is not None:
                action=np.asarray(control.get("action"),float)
                if (not np.isclose(control.get("decision_time_s",np.nan),t0,atol=1e-9) or
                        action.shape!=(self.defender_slots,3) or not np.isfinite(action).all() or
                        np.any((action<0)|(action>1))):
                    raise ValueError(f"frame {k} invalid step-start control")
                if "pairs" in control:
                    pairs={int(i):row for i,row in control["pairs"].items()}
                    if set(pairs)!={int(i) for i in before["assignment"]}:
                        raise ValueError(f"frame {k} invalid step-start pairs")
                    for i,row in pairs.items():
                        direction=np.asarray(row.get("guidance_direction_unit"),float)
                        if (direction.shape!=(3,) or not np.isfinite(direction).all() or
                                not np.isclose(np.linalg.norm(direction),1.,atol=1e-9) or
                                int(row.get("assigned_target",-1))!=int(before["assignment"][str(i)])):
                            raise ValueError(f"frame {k} invalid step-start pair {i}")
                else:
                    direction=np.asarray(control.get("guidance_direction_unit"),float)
                    if (direction.shape!=(3,) or not np.isfinite(direction).all() or
                            not np.isclose(np.linalg.norm(direction),1.,atol=1e-9) or
                            str(control.get("assigned_target"))!=str(before.get("assignment",{}).get("0",
                                before.get("assignment",{}).get(0)))):
                        raise ValueError(f"frame {k} invalid step-start control")
            state=combined(before)
            active=masks(before)
            executed=np.asarray(after.get("executed_velocities"),float)
            if executed.shape!=(self.defender_slots,3) or not np.isfinite(executed).all():
                raise ValueError(f"frame {k} invalid executed_velocities")
            for i in range(self.defender_slots):
                speed=np.linalg.norm(executed[i])
                if active[i] and not np.isclose(speed,self.config.defender_speeds[i],**tolerance):
                    raise ValueError(f"frame {k} defender {i} executed speed mismatch")
                if not active[i] and speed!=0:
                    raise ValueError(f"frame {k} inactive defender {i} has motion")
            inactive_at={}
            previous_time=t0
            for event in after["events"]:
                try:
                    time=float(event["time"])
                    members=event["members"]
                    positions=event["positions"]
                    states=event["states"]
                    kind=event["kind"]
                except (KeyError,TypeError) as exc:
                    raise ValueError(f"frame {k} invalid event field: {exc}") from exc
                if (not np.isfinite(time) or time<t0-1e-10 or time>t1+1e-10
                        or time<previous_time-1e-10):
                    raise ValueError(f"frame {k} event time outside step or out of order")
                previous_time=time
                if (type(members) is not list or any(type(m) is not int or m<0 or m>=self.defender_slots+self.intruder_slots for m in members)
                        or len(positions)!=len(members) or len(states)!=len(members)):
                    raise ValueError(f"frame {k} invalid event members or states")
                for member,position,recorded in zip(members,positions,states):
                    if not active[member]:
                        raise ValueError(f"frame {k} event uses inactive member {member}")
                    expected=state[member].copy()
                    if member<self.defender_slots:
                        expected[:3]+=executed[member]*(time-t0)
                        expected[3:]=executed[member]
                    else:
                        expected=self._intruder_at(member-self.defender_slots,time)
                    if kind=="member_inactive":
                        expected[3:]=0.
                    if (np.shape(position)!=(3,) or np.shape(recorded)!=(6,)
                            or not np.allclose(position,expected[:3],**tolerance)
                            or not np.allclose(recorded,expected,**tolerance)):
                        raise ValueError(f"frame {k} event state mismatch for member {member}")
                if kind=="member_inactive":
                    if len(members)!=1 or members[0] in inactive_at:
                        raise ValueError(f"frame {k} duplicate or invalid member exit")
                    member=members[0]
                    inactive_at[member]=np.asarray(states[0],float)
                    active[member]=False
                all_events.append(event)
            expected=state.copy()
            for member in range(self.defender_slots+self.intruder_slots):
                if member in inactive_at:
                    expected[member]=inactive_at[member]
                elif active[member]:
                    if member<self.defender_slots:
                        expected[member,:3]+=executed[member]*(t1-t0)
                        expected[member,3:]=executed[member]
                    else:
                        expected[member]=self._intruder_at(member-self.defender_slots,t1)
            if not np.allclose(expected,combined(after),**tolerance):
                raise ValueError(f"frame {k} reconstructed state mismatch")
            if active!=masks(after):
                raise ValueError(f"frame {k} activity mask mismatch")
        result=self.data.get("result")
        if (not isinstance(result,dict) or result.get("events")!=all_events
                or result.get("outcome")!=self.frames[-1]["outcome"]
                or not np.isclose(result.get("time",np.nan),self.duration,**tolerance)):
            raise ValueError("result events, outcome or time mismatch")

    def _intruder_at(self,index,time):
        points,distances,spec=self.references[index]
        distance=min(time*spec["speed"],distances[-1])
        segment=min(np.searchsorted(distances,distance,side="right")-1,len(points)-2)
        segment=max(segment,0)
        fraction=(distance-distances[segment])/(distances[segment+1]-distances[segment])
        position=points[segment]+fraction*(points[segment+1]-points[segment])
        velocity=(points[segment+1]-points[segment])/(distances[segment+1]-distances[segment])*spec["speed"]
        return np.r_[position,velocity]

    def at(self,time):
        if not np.isfinite(time) or not 0<=time<=self.duration:
            raise ValueError("time outside ideal trajectory")
        index=bisect_right(self.times,time)-1
        before=self.frames[index]
        exact=time==self.times[index]
        after=before if exact else self.frames[index+1]
        states=[np.asarray(before["defenders"][i],float).copy() for i in range(self.defender_slots)]
        states += [np.asarray(before["intruders"][j],float).copy() for j in range(self.intruder_slots)]
        masks=list(before["defender_mask"])+list(before["intruder_mask"])
        if not exact:
            executed=np.asarray(after["executed_velocities"],float)
            if executed.shape!=(self.defender_slots,3):
                raise ValueError("missing executed ideal motion")
            elapsed=time-before["time"]
            for i in range(self.defender_slots):
                if masks[i]:
                    states[i][:3]+=executed[i]*elapsed
                    states[i][3:]=executed[i]
            for j in range(len(self.references)):
                if masks[self.defender_slots+j]:
                    states[self.defender_slots+j]=self._intruder_at(j,time)
            for event in after["events"]:
                if event["time"]>time or event["kind"]!="member_inactive":
                    continue
                member=event["members"][0]
                if masks[member]:
                    states[member]=np.asarray(event["states"][0],float)
                    masks[member]=False
        entities=[]
        for member,state in enumerate(states):
            if member>=self.defender_slots and member-self.defender_slots>=len(self.references):
                continue
            if member<self.defender_slots and not self.frames[0]["defender_mask"][member]:
                continue
            entities.append(dict(id=member,name=f"D{member}" if member<self.defender_slots else f"I{member-self.defender_slots}",
                role="defender" if member<self.defender_slots else "intruder",position=state[:3].tolist(),
                velocity=state[3:].tolist(),active=bool(masks[member])))
        visible=[event for event in self.events if event["time"]<=time and event["kind"] in PHYSICAL_EVENTS]
        return dict(time=time,step=after["step"],outcome=after["outcome"] if exact else before["outcome"],
                    entities=entities,events=visible,state_basis="recorded" if exact else "ideal_motion")
