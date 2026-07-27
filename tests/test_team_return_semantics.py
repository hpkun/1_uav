import numpy as np
import pytest

from uav_env.algorithms.mappo.adapter import AdapterStep, SyncCombatVectorEnv
from uav_env.algorithms.mappo.config import load_mappo_config,validate_mappo_config
from uav_env.algorithms.mappo.metrics import combat_outcome_rates,evaluation_key
from uav_env.combat.events import EpisodeOutcome


def test_two_agent_step_and_multistep_return_semantics():
    steps = []
    for rewards in ([2.0, 4.0], [1.0, 3.0]):
        values = np.asarray(rewards, dtype=np.float32)
        steps.append(AdapterStep(np.zeros((2, 28), np.float32), np.zeros(40, np.float32), values,
                                 float(values.mean()), float(values.sum()), np.ones(2, np.float32),
                                 np.ones((2, 15), bool), False, False, {}))
    stacked = SyncCombatVectorEnv._stack(steps)
    assert np.allclose(stacked["team_rewards"], [3.0, 2.0])
    assert np.allclose(stacked["agent_reward_sums"], [6.0, 4.0])
    assert stacked["team_rewards"].sum() == 5.0
    assert stacked["agent_reward_sums"].sum() == 10.0


def test_one_agent_definitions_coincide():
    rewards = np.asarray([2.5], dtype=np.float32)
    step = AdapterStep(np.zeros((1, 11), np.float32), np.zeros(10, np.float32), rewards, 2.5, 2.5,
                       np.ones(1, np.float32), np.ones((1, 15), bool), False, False, {})
    assert step.team_reward == step.agent_reward_sum == float(step.agent_rewards[0])


def test_elimination_and_timeout_survival_wins_are_distinct():
 outcomes=[EpisodeOutcome("red",True,False,"blue_eliminated",10,5.),EpisodeOutcome("red",True,True,"timeout",400,200.),EpisodeOutcome("draw",True,True,"timeout",400,200.),EpisodeOutcome("blue",False,True,"red_eliminated",20,10.)]
 rates=combat_outcome_rates(outcomes)
 assert rates["overall_red_win_rate"]==.5
 assert rates["elimination_win_rate"]==rates["timeout_survival_win_rate"]==rates["decisive_win_rate"]==.25
 assert rates["timeout_rate"]==.5 and rates["draw_rate"]==.25


def test_combat_checkpoint_selection_prioritizes_elimination():
 base={"elimination_win_rate":0.,"timeout_rate":0.,"mean_effective_damage":300.,"overall_red_win_rate":1.,"mean_team_episode_return":100.}
 elimination={**base,"elimination_win_rate":.01,"timeout_rate":1.,"mean_effective_damage":0.,"overall_red_win_rate":.01,"mean_team_episode_return":-100.}
 assert evaluation_key(elimination,"combat")>evaluation_key(base,"combat")


def test_combat_checkpoint_selection_does_not_prefer_fast_failure_when_winless():
 fast_failure={"timeout_rate":0.0,"mean_red_survivors":0.0,"mean_blue_survivors":3.0,"mean_effective_damage":0.0,"mean_team_episode_return":-100.0}
 surviving_damage={"timeout_rate":1.0,"mean_red_survivors":2.0,"mean_blue_survivors":3.0,"mean_effective_damage":20.0,"mean_team_episode_return":-20.0}
 assert evaluation_key(surviving_damage,"combat")>evaluation_key(fast_failure,"combat")


def test_combat_checkpoint_selection_prioritizes_effective_damage_at_equal_wins():
 low_damage={"elimination_win_rate":0.0,"overall_red_win_rate":0.0,"mean_effective_damage":1.0,"mean_survivor_difference":0.0}
 high_damage={**low_damage,"mean_effective_damage":2.0,"timeout_rate":1.0}
 assert evaluation_key(high_damage,"combat")>evaluation_key(low_damage,"combat")


def test_combat_checkpoint_selection_prioritizes_survivor_difference_after_damage():
 lower_survival={"elimination_win_rate":0.0,"overall_red_win_rate":0.0,"mean_effective_damage":2.0,"mean_red_survivors":1.0,"mean_blue_survivors":3.0}
 higher_survival={**lower_survival,"mean_red_survivors":2.0}
 assert evaluation_key(higher_survival,"combat")>evaluation_key(lower_survival,"combat")


def test_combat_checkpoint_selection_accepts_legacy_evaluation_dict():
 assert evaluation_key({"overall_red_win_rate":0.0},"combat") == (0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0,-0.0)


def test_validation_and_test_seed_ranges_must_not_overlap():
 config=load_mappo_config("configs/mappo_smoke_1v1.yaml");config.update(validation_seed_start=100,validation_episodes=10,test_seed_start=105,test_episodes=10)
 with pytest.raises(ValueError,match="must not overlap"): validate_mappo_config(config)
