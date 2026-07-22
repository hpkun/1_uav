"""Rule-based opponent interfaces."""

from uav_env.opponents.base import RuleOpponent
from uav_env.opponents.predictive_rule import PredictiveRuleOpponent
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.straight import StraightOpponent

__all__ = ["PredictiveRuleOpponent", "PursuitOpponent", "RuleOpponent", "StraightOpponent"]
