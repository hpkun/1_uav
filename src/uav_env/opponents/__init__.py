"""Rule-based opponent interfaces."""

from uav_env.opponents.base import RuleOpponent
from uav_env.opponents.greedy_combat import GreedyCombatOpponent
from uav_env.opponents.predictive_rule import PredictiveRuleOpponent
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent

__all__ = ["GreedyCombatOpponent", "PredictiveRuleOpponent", "PursuitOpponent", "RandomOpponent", "RuleOpponent", "StraightOpponent"]
from uav_env.opponents.team_controller import TeamRuleController

__all__ = ["TeamRuleController"]
