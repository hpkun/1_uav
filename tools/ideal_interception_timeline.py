"""Continuous-time playback of recorded ideal point-mass motion."""
from bisect import bisect_right
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from env.ideal_interception.config import PROTOCOL
from .ideal_interception_trace import TRACE_VERSION

PHYSICAL_EVENTS = {"interception_contact","friendly_collision","intruder_collision",
                   "region_breach","boundary_exit","path_complete"}


class Timeline:
    def __init__(self,source):
        opener=gzip.open if Path(source).suffix==".gz" else open
        with opener(source,"rt",encoding="utf-8") as stream:
            data=json.load(stream)
        if data.get("trace_version")!=TRACE_VERSION or data.get("environment_protocol")!=PROTOCOL:
            raise ValueError("unsupported ideal trajectory protocol")
        self.data=data
        self.frames=data["frames"]
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
        for reference in data["references"]:
            points=np.asarray(reference["points"],float)
            spec=reference["specification"]
            if (points.ndim!=2 or points.shape[1]!=3 or len(points)!=spec["polyline_points"]
                    or not np.isfinite(points).all() or
                    hashlib.sha256(points.tobytes()).hexdigest()!=spec["polyline_sha256"]):
                raise ValueError("recorded ideal reference hash mismatch")
            distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            self.references.append((points,distances,spec))
        if len(self.references)!=sum(self.frames[0]["intruder_mask"]):
            raise ValueError("reference/entity count mismatch")

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
        states=[np.asarray(before["defenders"][i],float).copy() for i in range(5)]
        states += [np.asarray(before["intruders"][j],float).copy() for j in range(2)]
        masks=list(before["defender_mask"])+list(before["intruder_mask"])
        if not exact:
            executed=np.asarray(after["executed_velocities"],float)
            if executed.shape!=(5,3):
                raise ValueError("missing executed ideal motion")
            elapsed=time-before["time"]
            for i in range(5):
                if masks[i]:
                    states[i][:3]+=executed[i]*elapsed
                    states[i][3:]=executed[i]
            for j in range(len(self.references)):
                if masks[5+j]:
                    states[5+j]=self._intruder_at(j,time)
            for event in after["events"]:
                if event["time"]>time or event["kind"]!="member_inactive":
                    continue
                member=event["members"][0]
                if masks[member]:
                    states[member]=np.asarray(event["states"][0],float)
                    masks[member]=False
        entities=[]
        for member,state in enumerate(states):
            if member>=5 and member-5>=len(self.references):
                continue
            if member<5 and not self.frames[0]["defender_mask"][member]:
                continue
            entities.append(dict(id=member,name=f"D{member}" if member<5 else f"I{member-5}",
                role="defender" if member<5 else "intruder",position=state[:3].tolist(),
                velocity=state[3:].tolist(),active=bool(masks[member])))
        visible=[event for event in self.events if event["time"]<=time and event["kind"] in PHYSICAL_EVENTS]
        return dict(time=time,step=after["step"],outcome=after["outcome"] if exact else before["outcome"],
                    entities=entities,events=visible,state_basis="recorded" if exact else "ideal_motion")
