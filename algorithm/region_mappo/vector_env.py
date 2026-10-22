"""Region worker lifecycle; audited parent batching/reset/snapshot logic is reused."""
import os
import traceback
from algorithm.common.vector_env import ParallelVectorEnv
from .metrics import RegionAccumulator


def region_worker(pipe,config):
    from env.region_competition import RegionCompetitionEnv
    try:
        env=RegionCompetitionEnv(config)
        pipe.send(('ok',os.getpid()))
        while True:
            command,payload=pipe.recv()
            if command=='close': break
            if command=='reset': value=env.reset(payload)
            elif command=='step': value=env.step(payload)
            elif command=='snapshot': value=env.snapshot()
            elif command=='restore': env.restore(payload); value=None
            else: raise ValueError('unknown region worker command')
            pipe.send(('ok',value))
    except BaseException:
        try: pipe.send(('error',traceback.format_exc()))
        except (OSError,EOFError): pass
    finally:
        pipe.close()


class RegionVectorEnv(ParallelVectorEnv):
    worker_target=staticmethod(region_worker)
    tracker_factory=RegionAccumulator

    def __init__(self,config=None,*args,**kwargs):
        from env.region_competition import RegionConfig
        super().__init__(config or RegionConfig(),*args,**kwargs)
