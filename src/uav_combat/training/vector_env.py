"""Correctness-first synchronous vectorization with deterministic independent resets."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..environment.env import PaperUAVCombatEnv


@dataclass
class VectorStep:
    observations: np.ndarray
    transition_next_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict]
    alive_masks: np.ndarray
    next_alive_masks: np.ndarray


class SyncVectorEnv:
    def __init__(self,num_envs:int,config="configs/paper_environment.yaml",base_seed:int=0,seed_stride:int=100000)->None:
        if num_envs<=0: raise ValueError("num_envs must be positive")
        self.envs=[PaperUAVCombatEnv(config) for _ in range(num_envs)]; self.num_envs=num_envs; self.base_seed=int(base_seed); self.seed_stride=int(seed_stride)
        self.episode_indices=np.zeros(num_envs,np.int64); self.current_observations=np.zeros((num_envs,4,45),np.float32); self.current_alive_masks=np.ones((num_envs,4),np.float32); self.last_reset_seeds=np.zeros(num_envs,np.int64)

    def seed_for(self,env_id:int,episode_index:int)->int:
        return self.base_seed+env_id+episode_index*self.seed_stride

    def reset(self)->np.ndarray:
        rows=[]
        for i,env in enumerate(self.envs):
            seed=self.seed_for(i,int(self.episode_indices[i])); self.last_reset_seeds[i]=seed; rows.append(env.reset(seed)[0])
        self.current_observations=np.stack(rows); self.current_alive_masks=np.stack([e.red_alive_mask for e in self.envs]); return self.current_observations.copy()

    def step_batch(self,actions:np.ndarray,blue_actions:np.ndarray|None=None,auto_reset:bool=True)->VectorStep:
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(self.num_envs,4,3): raise ValueError("vector actions must be [env,4,3]")
        alive_before=self.current_alive_masks.copy(); results=[]
        for i,env in enumerate(self.envs): results.append(env.step(actions[i],None if blue_actions is None else blue_actions[i]))
        transition_next=np.stack([r[0] for r in results]); rewards=np.stack([r[1] for r in results]); terminated=np.asarray([r[2] for r in results]); truncated=np.asarray([r[3] for r in results]); infos=[r[4] for r in results]
        next_alive=np.stack([info["red_alive_mask"] for info in infos]); current=transition_next.copy(); current_alive=next_alive.copy()
        if auto_reset:
            for i,done in enumerate(terminated|truncated):
                if done:
                    self.episode_indices[i]+=1; seed=self.seed_for(i,int(self.episode_indices[i])); self.last_reset_seeds[i]=seed
                    current[i]=self.envs[i].reset(seed)[0]; current_alive[i]=self.envs[i].red_alive_mask
        self.current_observations=current; self.current_alive_masks=current_alive
        return VectorStep(current.copy(),transition_next,rewards,terminated,truncated,infos,alive_before,next_alive)

    def step(self,actions:np.ndarray):
        result=self.step_batch(actions)
        return result.observations,result.rewards,result.terminated,result.truncated,result.infos
