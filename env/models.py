"""Six dimensional flight states and independent inspection task state."""
from dataclasses import dataclass
import numpy as np


@dataclass
class TaskState:
    states: np.ndarray  # [N,6]: x,y,z,v,gamma,psi; z points up.
    waypoints: np.ndarray  # [M,3], index is immutable waypoint ID.
    priorities: np.ndarray
    completed: np.ndarray
    active_mask: np.ndarray


def pairwise_distances(states):
    p = states[:, :3]
    return np.linalg.norm(p[:, None] - p[None, :], axis=-1)
