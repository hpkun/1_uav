"""Read an existing trajectory without importing the simulator or loading a model.

Usage: python -m tools.trajectory_audit trajectory.json --output new_report.md
Output files must be new; the input and all original artifacts stay untouched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


COMPONENTS = ('progress', 'completion', 'separation', 'efficiency', 'terminal')


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def heading_error(state, point):
    delta = np.asarray(point) - np.asarray(state)[:3]
    return float(wrap(math.atan2(delta[1], delta[0]) - state[5]))


def describe(values):
    a = np.asarray(values, dtype=float)
    return 'missing' if not a.size else f'{a.min():.6g} / {a.mean():.6g} / {a.max():.6g}'


def audit(path, stride=25):
    """Return Markdown derived only from recorded values, with no writes or rollout."""
    if stride < 1:
        raise ValueError('stride must be positive')
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    frames = data['frames']
    if len(frames) < 2:
        raise ValueError('at least an initial frame and one transition are required')
    states = np.asarray([f['states'] for f in frames], dtype=float)
    points = np.asarray(frames[0]['waypoints'], dtype=float)
    if states.ndim != 3 or states.shape[-1] != 6 or points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError('expected states [T+1,N,6] and waypoints [M,3]')
    if not np.isfinite(states).all() or not np.isfinite(points).all():
        raise ValueError('nonfinite recorded state/waypoint')
    t, n, _ = states.shape
    config = data['config']
    dt = config['dt']
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('invalid recorded dt')
    controls = np.asarray([f['physical_actions'] for f in frames[1:]], dtype=float)
    masks = np.asarray([f.get('current_active_mask', frames[k]['active_mask'])
                        for k, f in enumerate(frames[1:])], dtype=bool)
    assignments = np.asarray([f.get('transition_assignments', frames[k]['assignments'])
                              for k, f in enumerate(frames[1:])], dtype=int)
    if controls.shape != (t-1, n, 3) or masks.shape != (t-1, n) or assignments.shape != masks.shape:
        raise ValueError('inconsistent transition dimensions')
    if not np.isfinite(controls[masks]).all():
        raise ValueError('nonfinite active physical control')
    if ((assignments < -1) | (assignments >= len(points))).any():
        raise ValueError('invalid assignment index')
    if any(not np.array_equal(f['waypoints'], points) for f in frames):
        raise ValueError('this auditor requires stationary, fixed-ID waypoints')
    prev_error = 0.
    prev_checks = 0
    reward_errors = dict.fromkeys(COMPONENTS + ('sum',), 0.)
    reward_checks = 0
    event_keys = set()
    duplicate_events = 0
    event_mismatches = completed_mismatches = 0
    for k, f in enumerate(frames[1:]):
        if f['episode_length'] != frames[k]['episode_length'] + 1:
            raise ValueError('nonconsecutive episode frames')
        if 'previous_states' in f:
            prev_checks += 1
            prev_error = max(prev_error, float(np.max(np.abs(np.asarray(f['previous_states']) - states[k]))))
        events = f.get('events', [])
        expected = {c: np.zeros(n) for c in COMPONENTS}
        active_next = np.asarray(f['active_mask'], dtype=bool)
        expected_events = []
        expected_completed = np.asarray(frames[k]['completed'], dtype=bool).copy()
        for j in np.flatnonzero(~expected_completed):
            distances = np.linalg.norm(states[k+1,:,:3]-points[j],axis=1)
            eligible = np.flatnonzero(active_next & (distances <= config['completion_radius']))
            if len(eligible):
                expected_events.append((int(j),int(eligible[0])))
                expected_completed[j] = True
        event_mismatches += int(sorted((e['waypoint_id'],e['uav_id']) for e in events) != expected_events
                                or any(e['step'] != f['episode_length'] for e in events))
        completed_mismatches += int(not np.array_equal(expected_completed,f['completed']))
        for i in range(n):
            j = assignments[k, i]
            if masks[k, i]:
                expected['efficiency'][i] = -.01
                if j >= 0:
                    expected['progress'][i] = .002 * (np.linalg.norm(points[j]-states[k, i, :3]) - np.linalg.norm(points[j]-states[k+1, i, :3]))
            others = np.flatnonzero(active_next & (np.arange(n) != i))
            if active_next[i]:
                if len(others):
                    d = np.linalg.norm(states[k+1, others, :3] - states[k+1, i, :3], axis=1)
                    expected['separation'][i] = -np.maximum(0, 1-d/config['safe_distance']).mean()
                if f['success']:
                    expected['terminal'][i] = 20 - f['episode_length']//25
        for e in events:
            key = e['waypoint_id']
            duplicate_events += int(key in event_keys)
            event_keys.add(key)
        for _, owner in expected_events:
            expected['completion'][owner] += 10
        if 'reward_components' in f and 'rewards' in f:
            reward_checks += n
            parts = f['reward_components']
            for c in COMPONENTS:
                actual = np.asarray(parts[c], dtype=float)
                if actual.shape != (n,) or not np.isfinite(actual).all():
                    raise ValueError('invalid reward component')
                reward_errors[c] = max(reward_errors[c], float(np.max(np.abs(actual-expected[c]))))
            rewards = np.asarray(f['rewards'], dtype=float)
            if rewards.shape != (n,) or not np.isfinite(rewards).all():
                raise ValueError('invalid reward')
            reward_errors['sum'] = max(reward_errors['sum'], float(np.max(np.abs(sum(np.asarray(parts[c]) for c in COMPONENTS)-rewards))))
    diagnostic_count = sum('policy_diagnostics' in f for f in frames[1:])
    if diagnostic_count not in (0,t-1):
        raise ValueError('partial policy diagnostics; compare only complete recordings')
    policy = alpha = beta = None
    if diagnostic_count:
        policy = np.asarray([f['policy_diagnostics']['normalized_actions'] for f in frames[1:]],dtype=float)
        alpha = np.asarray([f['policy_diagnostics']['beta_alpha'] for f in frames[1:]],dtype=float)
        beta = np.asarray([f['policy_diagnostics']['beta_beta'] for f in frames[1:]],dtype=float)
        if any(a.shape != controls.shape or not np.isfinite(a).all() for a in (policy,alpha,beta)):
            raise ValueError('invalid policy diagnostics dimensions or values')
        if ((policy<0)|(policy>1)).any() or (alpha<=0).any() or (beta<=0).any():
            raise ValueError('policy actions or Beta parameters out of range')
    lines = [f'# Recorded trajectory audit: {path.name}', '',
             f'- Source: `{path.resolve()}`', f'- SHA256: `{hashlib.sha256(raw).hexdigest()}`',
             f'- Environment: `{data.get("environment_version", "missing")}`; seed={data.get("seed")}; scale={config.get("scale")}.',
             f'- Provenance: `{json.dumps(data.get("provenance", {}), ensure_ascii=False)}`',
             f'- Steps={t-1}; dt={dt}s; elapsed={(t-1)*dt:g}s; success={frames[-1]["success"]}; completion={frames[-1]["completion_rate"]:.6g}.',
             f'- State/previous-state max error={prev_error:.9g} ({prev_checks} checked transitions); repeated waypoint events={duplicate_events}.',
             f'- Geometry/lowest-active-owner event mismatches={event_mismatches}; completed-flag mismatches={completed_mismatches}.',
             f'- Reward checks={reward_checks} UAV-transitions; max absolute residuals against current inspection rules: `{reward_errors}`.',
             '- Reward residuals test current code rules, not either combat paper. No input was simulated or changed.',
             '- Recorded physical controls are executed controls. Inverse mapping is NOT proof of a raw Beta sample; idle controls override policy actions.',
             (f'- Original policy actions and Beta parameters present for {diagnostic_count} transitions; selection={frames[1]["policy_diagnostics"]["selection"]}.' if diagnostic_count else '- Raw policy action/Beta alpha/Beta beta/log_probability/entropy are absent in this recording. No distribution inference is made.'),
             '- Saturation means executed normalized control <=0.01 or >=0.99 (1% of action range); a state at a limit is distinct from a clipping event.',
             '- Heading increments use wrap(delta psi); total turn sums absolute increments, net turn sums signed increments. Units below are degrees.', '']
    for i in range(n):
        s = states[:, i]
        valid = masks[:, i]
        u = controls[valid, i]
        inv = (u + [math.pi, 1, 0]) / [2*math.pi, 3, 8]
        assigned = valid & (assignments[:, i] >= 0)
        initial_j = frames[0]['assignments'][i]
        turns = np.degrees(wrap(np.diff(s[:, 5]))[valid])
        length = np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1)[valid].sum()
        er = [math.degrees(heading_error(states[k, i], points[assignments[k, i]]))
              for k in np.flatnonzero(assigned)]
        lines += [f'## UAV {i}', '', f'- Active transitions={int(valid.sum())}; assigned={int(assigned.sum())}; idle={int((valid & ~assigned).sum())}; path length={length:.3f}m.',
                  f'- Position start={s[0,:3].round(3).tolist()}; end={s[-1,:3].round(3).tolist()}.',
                  f'- Speed min/mean/max={describe(s[:,3])} m/s; gamma={describe(np.degrees(s[:,4]))} deg.',
                  f'- Heading start/end={math.degrees(s[0,5]):.3f}/{math.degrees(s[-1,5]):.3f}; net/total turn={turns.sum():.3f}/{np.abs(turns).sum():.3f} deg.',
                  f'- Assigned pre-action heading error min/mean/max={describe(er)} deg; mean absolute={np.mean(np.abs(er)) if er else float("nan"):.6g} deg.']
        if initial_j >= 0:
            d = np.linalg.norm(s[:, :3]-points[initial_j], axis=1)
            lines += [f'- Same initial target P{initial_j}: distance start/end/min={d[0]:.3f}/{d[-1]:.3f}/{d.min():.3f}m; heading error start/end={math.degrees(heading_error(s[0],points[initial_j])):.3f}/{math.degrees(heading_error(s[-1],points[initial_j])):.3f} deg.']
        lines += ['', '|Control|Physical min / mean / max|Executed normalized min / mean / max|Saturation %|', '|---|---|---|---|']
        for axis, label in enumerate(('bank mu (rad)', 'tangential load', 'normal load')):
            sat = 100*np.mean((inv[:,axis]<=.01)|(inv[:,axis]>=.99)) if len(u) else float('nan')
            lines.append(f'|{label}|{describe(u[:,axis])}|{describe(inv[:,axis])}|{sat:.3f}|')
        lines += ['', f'- Executed bank with cos(mu)<0: {100*np.mean(np.cos(u[:,0])<0) if len(u) else float("nan"):.3f}%.']
        if diagnostic_count:
            physical_candidate = policy[:,i]*[2*math.pi,3,8]+[-math.pi,-1,0]
            difference = np.abs(physical_candidate[assigned]-controls[assigned,i])
            lines += ['', f'- Policy -> executed physical control max residual on assigned active transitions: {difference.max() if difference.size else "missing"}. Idle overrides are excluded.',
                      '- Recorded policy diagnostics min / mean / max:', '',
                      '|Axis|Raw normalized action|Beta alpha|Beta beta|', '|---|---|---|---|']
            for j in range(3):
                lines.append(f'|{j}|{describe(policy[valid,i,j])}|{describe(alpha[valid,i,j])}|{describe(beta[valid,i,j])}|')
            for key in ('log_probability','entropy'):
                lines.append(f'- Recorded {key} (sum over three action dimensions): {describe([f["policy_diagnostics"][key][i] for k,f in enumerate(frames[1:]) if valid[k]])}.')
        for key in ('gamma_clips', 'speed_clips', 'nonfinite_states'):
            if all('numerical_events' in f for f in frames[1:]):
                count = sum(f['numerical_events'][key][i] for f in frames[1:])
                lines.append(f'- {key}: {count} events / {int(valid.sum())} active transitions.')
        lines += ['', '|Waypoint (fixed ID)|Distance start / end / closest (m)|Completed by episode end|', '|---|---|---|']
        for j, p in enumerate(points):
            d = np.linalg.norm(s[:,:3]-p, axis=1)
            lines.append(f'|P{j}|{d[0]:.3f} / {d[-1]:.3f} / {d.min():.3f}|{frames[-1]["completed"][j]}|')
        lines += ['', '|Reward component|Sum over this UAV|', '|---|---|']
        for c in COMPONENTS:
            lines.append(f'|{c}|{sum(f.get("reward_components",{}).get(c,[0]*n)[i] for f in frames[1:]):.9g}|')
        lines += ['', 'Sampled transition rows use PRE-action assignment/state/error, EXECUTED control, and POST-action reward/state. All switch/event rows are included.', '',
                  '|Step|Task before -> after|Pre distance / heading error|Post [x,y,z,v,gamma,psi]|Executed [mu,zeta,normal]|Reward|Events|', '|---|---|---|---|---|---|---|']
        for k, f in enumerate(frames[1:]):
            j = assignments[k, i]
            changed = j != f['assignments'][i]
            if k not in (0,t-2) and (k+1)%stride and not changed and not f.get('events') and not f.get('failures'):
                continue
            dist_error = 'idle'
            if j >= 0:
                dist_error = f'{np.linalg.norm(states[k,i,:3]-points[j]):.3f}m / {math.degrees(heading_error(states[k,i],points[j])):.3f}deg'
            after = states[k+1,i].copy(); after[4:] = np.degrees(after[4:])
            lines.append(f'|{f["episode_length"]}|{j} -> {f["assignments"][i]}|{dist_error}|{after.round(4).tolist()}|{controls[k,i].round(6).tolist()}|{f.get("rewards",[None]*n)[i]}|{f.get("events",[])} {f.get("failures",[])}|')
        lines.append('')
    lines += ['## Limits of inference', '', 'A persistent turn or large flight-path angle is a measured behavior, not proof of its learning cause. Different seeds, episode limits, failures or model versions cannot establish a controlled policy comparison. A recorded pose at the gamma/speed bound does not establish what an unconstrained integrator would have done. Missing fields remain missing.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--stride', type=int, default=25)
    args = p.parse_args()
    report = audit(args.trajectory, args.stride)
    if args.output:
        if args.output.resolve() == args.trajectory.resolve():
            raise ValueError('output cannot be the source trajectory')
        with args.output.open('x', encoding='utf-8') as f:
            f.write(report)
    else:
        print(report, end='')


if __name__ == '__main__':
    main()
