"""Table I and Eqs.(21)-(23), aggregate-sum 80/20 Blue opponent."""
import numpy as np
from env.integrator import rk4
from .geometry import pair_advantages

# Original table columns: mu, normal load, tangential load.
TABLE_I = np.array([[0,1,0],[0,1,2],[0,0,-1],
                    [-np.pi/3,8,0],[-np.pi/3,8,2],[-np.pi/3,8,-1],
                    [np.pi/3,8,0],[np.pi/3,8,2],[np.pi/3,8,-1],
                    [0,8,0],[0,8,2],[0,8,-1],
                    [np.pi,8,0],[np.pi,8,2],[np.pi,8,-1]],dtype=float)
BLUE_CONTROLS = TABLE_I[:,[0,2,1]].copy()


def candidate_scores(blue_state, red_states, config):
    """Pure virtual evaluation; no live state, counters, dwell or RNG mutation."""
    scores = np.full(15,-np.inf)
    for k,control in enumerate(BLUE_CONTROLS):
        counters = dict(speed_clips=0,nonfinite_states=0,integration_failures=0)
        candidate,ok = rk4(blue_state,control,config,counters)
        if ok:
            g, offense, defense, _ = pair_advantages(candidate[None,:],red_states,config)
            speed = (candidate[3]-config.v_min)/(config.v_max-config.v_min)
            scores[k] = np.sum(speed*g.cos_alpha_i+offense-defense.T)
    return scores


def choose_blue_controls(state, config, rng):
    controls = np.zeros((len(state.adversarial_states),3))
    choices = np.full(len(controls),-1,dtype=int)
    for j in np.flatnonzero(state.adversarial_alive):
        scores = candidate_scores(state.adversarial_states[j],state.friendly_states[state.friendly_alive],config)
        order = np.argsort(-scores,kind='stable')
        if np.isfinite(scores).sum()<2:
            raise FloatingPointError(f'Blue {j} has fewer than two feasible maneuvers')
        chosen = order[0 if rng.random()<config.opponent_best_probability else 1]
        controls[j],choices[j] = BLUE_CONTROLS[chosen],chosen
    return controls, choices
