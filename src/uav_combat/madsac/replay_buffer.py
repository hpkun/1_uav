"""Chunked fixed-slot CTDE replay buffer."""
from __future__ import annotations
import numpy as np
import torch


class ReplayBuffer:
    KEYS = ("observations","actions","rewards","next_observations","alive_masks","next_alive_masks","dones")

    def __init__(self, capacity: int = 1_000_000, num_agents: int = 4, observation_dim: int = 52, action_dim: int = 3, chunk_size: int = 4096) -> None:
        if capacity <= 0: raise ValueError("capacity must be positive")
        self.capacity,self.num_agents=int(capacity),int(num_agents); self.observation_dim,self.action_dim,self.chunk_size=observation_dim,action_dim,chunk_size
        self._chunks: dict[int,dict[str,np.ndarray]]={}; self.position=self.size=0

    def _chunk(self,index:int)->tuple[dict[str,np.ndarray],int]:
        ci,offset=divmod(index,self.chunk_size)
        if ci not in self._chunks:
            count=min(self.chunk_size,self.capacity-ci*self.chunk_size)
            self._chunks[ci]={
                "observations":np.empty((count,self.num_agents,self.observation_dim),np.float32),"actions":np.empty((count,self.num_agents,self.action_dim),np.float32),
                "rewards":np.empty((count,self.num_agents),np.float32),"next_observations":np.empty((count,self.num_agents,self.observation_dim),np.float32),
                "alive_masks":np.empty((count,self.num_agents),np.float32),"next_alive_masks":np.empty((count,self.num_agents),np.float32),"dones":np.empty((count,1),np.float32)}
        return self._chunks[ci],offset

    def push(self,observations,actions,rewards,next_observations,done:bool,alive_masks=None,next_alive_masks=None)->None:
        alive=np.ones(self.num_agents,np.float32) if alive_masks is None else np.asarray(alive_masks,np.float32)
        next_alive=np.ones(self.num_agents,np.float32) if next_alive_masks is None else np.asarray(next_alive_masks,np.float32)
        arrays=[np.asarray(x,np.float32) for x in (observations,actions,rewards,next_observations,alive,next_alive)]
        expected=((self.num_agents,self.observation_dim),(self.num_agents,self.action_dim),(self.num_agents,),(self.num_agents,self.observation_dim),(self.num_agents,),(self.num_agents,))
        if tuple(a.shape for a in arrays)!=expected: raise ValueError(f"invalid replay transition shapes: {[a.shape for a in arrays]}")
        arrays[1]=arrays[1]*arrays[4][:,None]
        chunk,offset=self._chunk(self.position)
        for key,value in zip(self.KEYS[:-1],arrays): chunk[key][offset]=value
        chunk["dones"][offset,0]=float(done); self.position=(self.position+1)%self.capacity; self.size=min(self.size+1,self.capacity)

    def push_batch(self,observations,actions,rewards,next_observations,dones,alive_masks,next_alive_masks)->None:
        observations=np.asarray(observations,np.float32); actions=np.asarray(actions,np.float32); rewards=np.asarray(rewards,np.float32); next_observations=np.asarray(next_observations,np.float32); alive_masks=np.asarray(alive_masks,np.float32); next_alive_masks=np.asarray(next_alive_masks,np.float32); dones=np.asarray(dones,np.float32).reshape(-1,1)
        n=observations.shape[0]
        expected=((n,self.num_agents,self.observation_dim),(n,self.num_agents,self.action_dim),(n,self.num_agents),(n,self.num_agents,self.observation_dim),(n,1),(n,self.num_agents),(n,self.num_agents))
        if tuple(a.shape for a in (observations,actions,rewards,next_observations,dones,alive_masks,next_alive_masks))!=expected: raise ValueError("invalid replay batch shapes")
        actions=actions*alive_masks[...,None]; data=(observations,actions,rewards,next_observations,alive_masks,next_alive_masks,dones); start=0
        while start<n:
            chunk,offset=self._chunk(self.position); count=min(n-start,len(chunk["dones"])-offset)
            for key,value in zip(self.KEYS,data): chunk[key][offset:offset+count]=value[start:start+count]
            self.position=(self.position+count)%self.capacity; self.size=min(self.size+count,self.capacity); start+=count

    def _rows(self,indices)->dict[str,np.ndarray]:
        rows=[self._chunk(int(i)) for i in indices]
        return {key:np.stack([chunk[key][offset] for chunk,offset in rows]) for key in self.KEYS}

    def sample(self,batch_size:int,rng:np.random.Generator|None=None,device:str|torch.device="cpu")->dict[str,torch.Tensor]:
        if batch_size<=0 or self.size<batch_size: raise ValueError("not enough transitions")
        indices=(rng or np.random.default_rng()).choice(self.size,size=batch_size,replace=False)
        return {key:torch.as_tensor(value,device=device) for key,value in self._rows(indices).items()}
