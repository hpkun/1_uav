"""Table I and Eqs.(21)-(23), aggregate-sum 80/20 Blue opponent."""
import numpy as np
from .framed_dynamics import advance, initial_frame, SimulationFault
from .geometry import pair_advantages

# Original table columns: mu, normal load, tangential load.
TABLE_I = np.array([[0,1,0],[0,1,2],[0,0,-1],
                    [-np.pi/3,8,0],[-np.pi/3,8,2],[-np.pi/3,8,-1],
                    [np.pi/3,8,0],[np.pi/3,8,2],[np.pi/3,8,-1],
                    [0,8,0],[0,8,2],[0,8,-1],
                    [np.pi,8,0],[np.pi,8,2],[np.pi,8,-1]],dtype=float)
BLUE_CONTROLS = TABLE_I[:,[0,2,1]].copy()


def aggregate_scores(pair_scores, mode='aggregate-sum'):
    """Eq.(22)/(24), Blue listed first; variants only for post-training tests."""
    if mode=='aggregate-sum': return np.sum(pair_scores)
    if mode=='opportunistic-max': return np.max(pair_scores)
    if mode=='threat-aware-maximin': return np.min(pair_scores)
    raise ValueError('unknown Blue aggregation')


def candidate_scores(blue_state, red_states, config, frame=None):
    """Pure virtual evaluation; no live state, counters, dwell or RNG mutation."""
    scores = np.full(15,-np.inf)
    for k,control in enumerate(BLUE_CONTROLS):
        try:
            candidate,_,_ = advance(blue_state,control,initial_frame(blue_state) if frame is None else frame,config)
        except SimulationFault:
            # Preserve the historical angular baseline's infeasible-candidate
            # scoring. In the default frame mode every true fault propagates.
            if config.integrator=='angle_rk4':continue
            raise
        g, offense, defense, _ = pair_advantages(candidate[None,:],red_states,config)
        speed = (candidate[3]-config.v_min)/(config.v_max-config.v_min)
        scores[k] = aggregate_scores(speed*g.cos_alpha_i+offense-defense.T,config.opponent_aggregation)
    return scores


def choose_blue_controls(state, config, rng):
    controls = np.zeros((len(state.adversarial_states),3))
    choices = np.full(len(controls),-1,dtype=int)
    for j in np.flatnonzero(state.adversarial_alive):
        frame = None if state.adversarial_frames is None else state.adversarial_frames[j]
        scores = candidate_scores(state.adversarial_states[j],state.friendly_states[state.friendly_alive],config,frame)
        order = np.argsort(-scores,kind='stable')
        if np.isfinite(scores).sum()<2:
            raise FloatingPointError(f'Blue {j} has fewer than two feasible maneuvers')
        chosen = order[0 if rng.random()<config.opponent_best_probability else 1]
        controls[j],choices[j] = BLUE_CONTROLS[chosen],chosen
    return controls, choices
