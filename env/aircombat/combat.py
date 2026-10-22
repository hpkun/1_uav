"""Simultaneous pairwise dwell events, independent of reward and APH."""
import numpy as np


def resolve_attacks(state, red_advantage, blue_advantage, dwell_steps):
    live = state.friendly_alive[:,None] & state.adversarial_alive[None,:]
    state.friendly_dwell = np.where(live & (red_advantage>0),state.friendly_dwell+1,0)
    state.adversarial_dwell = np.where(live.T & (blue_advantage>0),state.adversarial_dwell+1,0)
    # Compute BOTH event sets before touching any alive flags.
    events = []
    for side, counts, victims in (('Red',state.friendly_dwell,state.adversarial_alive),
                                  ('Blue',state.adversarial_dwell,state.friendly_alive)):
        for target in np.flatnonzero(victims):
            attackers = np.flatnonzero(counts[:,target]>=dwell_steps)
            if len(attackers):
                events.append(dict(attacker_side=side,attacker_id=int(attackers[0]),
                                   target_id=int(target),step=state.step_count,
                                   eligible_attackers=attackers.tolist()))
    for event in events:
        (state.adversarial_alive if event['attacker_side']=='Red' else state.friendly_alive)[event['target_id']] = False
    live = state.friendly_alive[:,None] & state.adversarial_alive[None,:]
    state.friendly_dwell *= live
    state.adversarial_dwell *= live.T
    state.events.extend(events)
    return events
