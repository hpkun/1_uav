"""Vectorized training and evaluation interfaces."""
from .vector_env import SyncVectorEnv
from .evaluator import evaluate
__all__ = ["SyncVectorEnv", "evaluate"]
