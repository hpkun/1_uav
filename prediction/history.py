"""Independent, timestamped histories for up to two intruders."""
from collections import deque
import math
import numpy as np

from .three_point import predict_two_steps


class IntruderHistory:
    def __init__(self, dt, slots=2):
        if not math.isfinite(dt) or dt <= 0 or type(slots) is not int or slots < 1:
            raise ValueError("invalid observation interval or slot count")
        self.dt, self.slots = float(dt), slots
        self.reset()

    def reset(self, scenario_id=None):
        self.scenario_id = scenario_id
        self.positions = [deque(maxlen=3) for _ in range(self.slots)]
        self.times = [deque(maxlen=3) for _ in range(self.slots)]
        self.active = np.zeros(self.slots, dtype=bool)
        self.retired = np.zeros(self.slots, dtype=bool)

    def observe(self, time, positions, mask, scenario_id=None):
        values = np.asarray(positions, dtype=np.float64)
        flags = np.asarray(mask)
        if (not math.isfinite(time) or time < 0 or values.shape != (self.slots, 3)
                or not np.isfinite(values).all() or flags.shape != (self.slots,)
                or flags.dtype.kind != "b"):
            raise ValueError("invalid intruder observation")
        if scenario_id != self.scenario_id:
            self.reset(scenario_id)
        for j in range(self.slots):
            if not flags[j]:
                if self.active[j]:
                    self.retired[j] = True
                self.positions[j].clear()
                self.times[j].clear()
                self.active[j] = False
                continue
            if self.retired[j]:
                raise ValueError(f"I{j} reactivated within one scenario")
            if self.times[j]:
                gap = time - self.times[j][-1]
                if gap <= 0:
                    raise ValueError(f"nonincreasing timestamp for I{j}")
                if not np.isclose(gap, self.dt, rtol=1e-8, atol=1e-9):
                    self.positions[j].clear()
                    self.times[j].clear()
            self.positions[j].append(values[j].copy())
            self.times[j].append(float(time))
            self.active[j] = True

    def status(self, target_index):
        if not 0 <= target_index < self.slots:
            raise IndexError(target_index)
        if not self.active[target_index]:
            return "INACTIVE"
        return "READY" if len(self.positions[target_index]) == 3 else "NOT_READY"

    def method_status(self, target_index, method):
        if method not in ("zuo_linear", "tan_quadratic"):
            raise ValueError("unknown prediction method")
        self.status(target_index)
        if not self.active[target_index]:
            return "INACTIVE"
        required = 2 if method == "zuo_linear" else 3
        return "READY" if len(self.positions[target_index]) >= required else "INSUFFICIENT_HISTORY"

    def predict(self, target_index):
        if self.status(target_index) != "READY":
            return None
        return predict_two_steps(np.stack(self.positions[target_index]), self.dt)

    def observed(self, target_index):
        """Return independent copies of this target's real, timestamped observations."""
        self.status(target_index)
        return (np.asarray(self.times[target_index], dtype=np.float64).copy(),
                np.asarray(self.positions[target_index], dtype=np.float64).reshape(-1, 3).copy())
