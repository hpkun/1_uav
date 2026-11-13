"""Read-only reset/diagnostic timing for the ideal environment."""
from dataclasses import replace
from time import perf_counter

from env.ideal_interception import InterceptionConfig,InterceptionEnv
from env.ideal_interception.scenario import diagnostics


def measure(repeats=30):
    for mode in ("fixed","random"):
        config=replace(InterceptionConfig(scale="5v2",intruder_mode="bezier"),initialization=mode)
        env=InterceptionEnv(config)
        started=perf_counter()
        for seed in range(repeats):
            env.reset(seed)
        reset_time=(perf_counter()-started)/repeats
        started=perf_counter()
        for _ in range(repeats):
            diagnostics(config,env.paths)
        diagnostic_time=(perf_counter()-started)/repeats
        print(f"{mode}: reset={reset_time*1000:.3f} ms | diagnostics={diagnostic_time*1000:.3f} ms | "
              f"diagnostic/reset={diagnostic_time/reset_time:.1%}")


if __name__=="__main__":
    measure()
