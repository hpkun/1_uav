from test_multi_observation import aircraft
from uav_env.combat.multi_combat import assign_targets
from uav_env.entities.type_profiles import UAVTypeProfile


def test_target_assignment_prefers_unique_and_reassigns(profile: UAVTypeProfile) -> None:
    reds,blues=aircraft(profile)
    assignments=assign_targets(blues,reds)
    assert len({a.target_id for a in assignments})==2
    reds[0].state.alive=False; reds[0].state.damaged=True
    reassigned=assign_targets(blues,reds)
    assert {a.target_id for a in reassigned}=={"red_1"}
