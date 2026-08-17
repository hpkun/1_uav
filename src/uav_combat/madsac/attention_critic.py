"""Independent centralized critic implementing Equations (16)-(17)."""
from __future__ import annotations
import math
import torch
from torch import nn


class AttentionCritic(nn.Module):
    """Compute one Q value per agent using explicit Wq/Wk/Wv multi-head attention."""
    def __init__(self, observation_dim: int = 45, action_dim: int = 3, hidden_dim: int = 256, attention_heads: int = 2, activation: str = "leaky_relu") -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim, self.attention_heads = hidden_dim, attention_heads
        self.head_dim = hidden_dim // attention_heads
        activation_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}.get(activation)
        if activation_cls is None: raise ValueError(f"unsupported critic activation: {activation}")
        self.embedding = nn.Sequential(nn.Linear(observation_dim + action_dim, hidden_dim), activation_cls(), nn.Linear(hidden_dim, hidden_dim), activation_cls())
        self.wq, self.wk, self.wv = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.q_network = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), activation_cls(), nn.Linear(hidden_dim, hidden_dim), activation_cls(), nn.Linear(hidden_dim, 1))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, alive_mask: torch.Tensor | None = None, return_attention: bool = False):
        if observations.ndim != 3 or actions.ndim != 3 or observations.shape[:2] != actions.shape[:2]:
            raise ValueError("critic inputs must be [batch, agents, features]")
        embedding = self.embedding(torch.cat([observations, actions], dim=-1)); b, n, _ = embedding.shape
        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(b, n, self.attention_heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.wq(embedding)), heads(self.wk(embedding)), heads(self.wv(embedding))
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if alive_mask is None: alive_mask = torch.ones((b,n),dtype=logits.dtype,device=logits.device)
        alive = alive_mask > 0.5
        valid = alive[:,None,None,:].expand(b,self.attention_heads,n,n) & ~torch.eye(n,dtype=torch.bool,device=logits.device).view(1,1,n,n)
        safe_logits = logits.masked_fill(~valid, -1e9)
        weights = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1,keepdim=True).clamp_min(1e-12)
        weights = weights * alive[:,None,:,None].to(logits.dtype)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, n, self.hidden_dim)
        values = self.q_network(torch.cat([embedding, context], dim=-1)).squeeze(-1)
        return (values, weights) if return_attention else values
