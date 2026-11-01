"""Read-only combat control/flight audit. Never trains or writes a checkpoint."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable
from algorithm.experiments import load_model, configurations, write_json
from algorithm.mappo.networks import normalize_inputs
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from env.aircombat import AirCombatEnv
from env.aircombat.framed_dynamics import frame_rhs, initial_frame, advance, SimulationFault
from env.aircombat.geometry import relative_geometry, pair_advantages
from env.dynamics import physical_actions, velocity_vector, wrap_angle
from env.integrator import rk4
from tools.diagnose_headon import transition_samples


PREFERRED = {
    'paper_raw': ('outputs/xu_iv_v2_2500/main_3v3/seed_1/latest.pt', 'main_3v3'),
    'fixed_norm': ('outputs/headon_norm_2500/headon_xu_norm_3v3/seed_1/latest.pt', 'headon_xu_norm_3v3'),
    'signed': ('outputs/headon_norm_signed_2500/headon_xu_norm_signed_3v3/seed_1/latest.pt', 'headon_xu_norm_signed_3v3'),
    'signed_gated': ('outputs/headon_norm_signed_gated_2500/headon_xu_norm_signed_gated_3v3/seed_1/latest.pt', 'headon_xu_norm_signed_gated_3v3'),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def discover():
    """Do not substitute a smoke checkpoint for a missing trained policy."""
    result = []
    for label, (filename, experiment) in PREFERRED.items():
        path = Path(filename)
        item = dict(label=label, checkpoint=str(path.resolve()), experiment=experiment)
        if not path.is_file():
            item.update(status='missing')
        else:
            try:
                trainer, c, payload = load_model(path)
                target = payload['train_config']['training_episodes']
                ec, tc = configurations(experiment, trainer.config.training_seed_index, episodes=target)
                if c != ec or trainer.config != tc or payload['training_episode_count'] != target:
                    raise ValueError('not a completed canonical experiment checkpoint')
                item.update(status='ready', sha256=digest(path), environment=c.to_dict(),
                            training=trainer.config.to_dict(), environment_version=payload['environment_version'],
                            completed_episodes=payload['training_episode_count'], updates=payload['updates'],
                            sampled_steps=payload['sampled_steps'])
                del trainer, payload
            except (ValueError, RuntimeError, KeyError) as exc:
                item.update(status='rejected', reason=str(exc))
        result.append(item)
    return result


def actor_step(trainer, observations, masks, deterministic):
    """Observe the REAL sample_actions calls with passive hooks; sample only once."""
    captured = {}
    def input_hook(module, values):
        captured['network_input'] = values[0].detach().cpu().numpy().copy()
    def output_hook(module, values, output):
        captured['parameters'] = (torch.nn.functional.softplus(output)+1).detach().cpu().numpy().copy()
    h1 = trainer.actor.network[0].register_forward_pre_hook(input_hook)
    h2 = trainer.actor.network.register_forward_hook(output_hook)
    try:
        actions, logp = trainer.sample_actions(observations, masks, deterministic=deterministic)
    finally:
        h1.remove(); h2.remove()
    a, b = np.split(captured.pop('parameters'), 2, axis=-1)
    mean = a/(a+b)
    captured.update(beta_alpha=a, beta_beta=b, beta_mean=mean,
                    beta_std=np.sqrt(a*b/((a+b)**2*(a+b+1))),
                    selection='deterministic_beta_mean' if deterministic else 'fresh_evaluation_beta_sample',
                    actual_actions=actions.cpu().numpy(), log_probability=logp.cpu().numpy(),
                    inactive_parameter_rows='network forward on zero slots, ignored by action mask')
    return captured


def scalar_rhs(state, control, gravity):
    """Independent direct transcription, not a finite difference."""
    x, y, z, v, gamma, psi = state
    mu, zeta, normal = control
    return np.array([v*np.cos(gamma)*np.cos(psi), v*np.cos(gamma)*np.sin(psi), v*np.sin(gamma),
        gravity*(zeta-np.sin(gamma)), gravity/v*(normal*np.cos(mu)-np.cos(gamma)),
        gravity*normal*np.sin(mu)/(v*np.cos(gamma))])


def motion_probe(state, following, frame, control, config):
    """Copies only: compare actual frame, scalar angle reference, chart reset CF."""
    h = float(np.linalg.norm(frame[:2, 0]))
    nominal = initial_frame(state)
    gauge = float(np.arctan2(frame[:,1]@nominal[:,2], frame[:,1]@nominal[:,1]))
    rhs = frame_rhs(np.r_[state[:4], frame.ravel()], control, config)
    # The derivative of v*u is dv*u + v*du. No wrapping/differencing here.
    vector_acceleration = rhs[3]*frame[:,0]+state[3]*rhs[4:].reshape(3,3)[:,0]
    safe = h > config.pole_cap_ratio+config.dt*config.gravity*(control[2]+1)/state[3]+.02
    item = dict(horizontal_ratio=h, gauge_angle_rad=gauge, effective_bank_rad=float(wrap_angle(control[0]+gauge)),
                frame_rhs=rhs, vector_acceleration=vector_acceleration, safe_angle_step=bool(safe))
    if h > .15:
        derivative = scalar_rhs(state, control, config.gravity)
        physical = np.array([rhs[3], config.gravity/state[3]*(control[2]*np.cos(control[0]+gauge)-np.cos(state[4])),
                            config.gravity*control[2]*np.sin(control[0]+gauge)/(state[3]*np.cos(state[4]))])
        item.update(paper_rhs=derivative, frame_speed_pitch_yaw_rhs=physical,
                    same_mu_rhs_difference=physical-derivative[3:])
    if safe:
        counters = dict(speed_clips=0, nonfinite_states=0, integration_failures=0)
        reference, ok = rk4(state.copy(), np.array(control), config, counters)
        item['angle_reference_accepted'] = bool(ok)
        if ok:
            difference = following-reference
            difference[5] = wrap_angle(difference[5])
            item['angle_rk4_difference'] = difference
            item['angle_rk4_position_error_m'] = float(np.linalg.norm(difference[:3]))
    # A counterfactual with reset local axes is NEVER fed back to the rollout.
    cf, _, _ = advance(state.copy(), np.array(control), nominal.copy(), config)
    item['counterfactual_chart_reset_position_difference_m'] = float(np.linalg.norm(following[:3]-cf[:3]))
    item['counterfactual_chart_reset_velocity_difference_m_s'] = float(np.linalg.norm(velocity_vector(following)-velocity_vector(cf)))
    return item


def episode(trainer, config, seed, scenario, policy, trace, *, deterministic=True, inspect=True):
    env = AirCombatEnv(config)
    obs, info = env.reset(seed, {'scenario': scenario})
    torch.manual_seed(seed+20000000)  # evaluation-only torch RNG, never env RNG
    initial = dict(red=env.state.friendly_states.copy(), blue=env.state.adversarial_states.copy())
    first = {side: dict(range=None, own_cone=None, eligible=None, kill=None) for side in ('Red','Blue')}
    counts = {side: dict(range_pairs=0, cone_pairs=0, aspect_veto_pairs=0, eligible_pairs=0,
                        two_step_pairs=0, interrupted_pairs=0) for side in first}
    all_events = []; switches = 0; step = 0; failure = None
    with gzip.open(trace, 'wt', encoding='utf-8', compresslevel=3) as stream:
        while not env.done:
            s = env.state
            r0, b0 = s.friendly_states.copy(), s.adversarial_states.copy()
            lr, lb, assigned = s.friendly_alive.copy(), s.adversarial_alive.copy(), s.assignments.copy()
            rf, bf = s.friendly_frames.copy(), s.adversarial_frames.copy()
            dwells = (s.friendly_dwell.copy(), s.adversarial_dwell.copy())
            if policy == 'trim':
                actions = np.tile([.5, 1/3, 1/8], (config.num_uavs,1))
                actions[~lr] = 0
                actor = dict(selection='fixed_trim_mu0_zeta0_n1', actual_actions=actions)
            else:
                actor = actor_step(trainer, obs, lr, deterministic)
                actions = actor['actual_actions']
            pre_obs = obs.copy(); central = info['centralized_state'].copy()
            try:
                obs, reward, term, trunc, info = env.step(actions)
            except SimulationFault as exc:
                failure = dict(reason=str(exc), step=step, controls=physical_actions(actions[lr]))
                stream.write(json.dumps(serializable(dict(failure=failure)), allow_nan=True)+'\n')
                break
            s = env.state; step = s.step_count
            r1, b1 = s.friendly_states, s.adversarial_states
            g, ra, ba, _ = pair_advantages(r1,b1,config)
            gb = relative_geometry(b1,r1,config.geometry_epsilon)
            pair_live = lr[:,None]&lb[None,:]
            ready_events = []; streaks = []; qualifies = []
            for side, geometry, live, old_dwell, advantage in (
                    ('Red',g,pair_live,dwells[0],ra), ('Blue',gb,pair_live.T,dwells[1],ba)):
                in_range = live&(geometry.distance>config.geometry_epsilon)&(geometry.distance<=config.tau_max)
                cone = in_range&(geometry.alpha_i<=config.alpha_max)
                eligible = cone&(geometry.alpha_j<np.pi/2)
                np.testing.assert_array_equal(eligible, live&(advantage>0))
                streak = np.where(eligible,old_dwell+1,0)
                ready = streak>=config.attack_dwell_steps
                for name, mask in (('range_pairs',in_range),('cone_pairs',cone),('aspect_veto_pairs',cone&~eligible),
                                   ('eligible_pairs',eligible),('two_step_pairs',ready),
                                   ('interrupted_pairs',(old_dwell>0)&live&~eligible)):
                    counts[side][name] += int(mask.sum())
                for key, mask in (('range',in_range),('own_cone',cone),('eligible',eligible)):
                    if first[side][key] is None and mask.any(): first[side][key]=step
                for victim in np.flatnonzero(ready.any(0)):
                    ready_events.append((side,int(np.flatnonzero(ready[:,victim])[0]),int(victim)))
                qualifies.append(eligible); streaks.append(streak)
            assert ready_events == [(e['attacker_side'],e['attacker_id'],e['target_id']) for e in info['events']]
            surviving = s.friendly_alive[:,None]&s.adversarial_alive[None,:]
            np.testing.assert_array_equal(s.friendly_dwell,streaks[0]*surviving)
            np.testing.assert_array_equal(s.adversarial_dwell,streaks[1]*surviving.T)
            np.testing.assert_array_equal(info['red_controls'][lr], physical_actions(actions[lr]))
            np.testing.assert_allclose(sum(info['reward_components'].values()),reward,rtol=0,atol=0)
            np.testing.assert_array_equal(r1[~lr],r0[~lr]); np.testing.assert_array_equal(b1[~lb],b0[~lb])
            event_rows = []
            for e in info['events']:
                side, i, j = e['attacker_side'], e['attacker_id'], e['target_id']
                own, target, geom = (r1,b1,g) if side=='Red' else (b1,r1,gb)
                event = dict(**e,time_s=step*config.dt,attacker_position=own[i,:3],target_position=target[j,:3],
                             distance=float(geom.distance[i,j]),alpha_i_deg=float(np.degrees(geom.alpha_i[i,j])),
                             alpha_j_deg=float(np.degrees(geom.alpha_j[i,j])))
                event_rows.append(event); all_events.append(event)
                if first[side]['kill'] is None: first[side]['kill']=step
            switches += int(((assigned!=s.assignments)&lr&s.friendly_alive).sum())
            row = dict(seed=seed,scenario=scenario,policy=policy,step=step,time_s=step*config.dt,
                pre_red=r0,pre_blue=b0,post_red=r1,post_blue=b1,pre_red_alive=lr,pre_blue_alive=lb,
                post_red_alive=s.friendly_alive,post_blue_alive=s.adversarial_alive,
                pre_assignments=assigned,post_assignments=s.assignments,raw_observation=pre_obs,
                raw_critic_input=central,actor=actor,red_actions=actions,red_controls=info['red_controls'],
                blue_controls=info['blue_controls'],blue_action_indices=info['blue_action_indices'],
                pre_red_frames=rf,pre_blue_frames=bf,post_red_frames=s.friendly_frames,post_blue_frames=s.adversarial_frames,
                reward=reward,reward_components=info['reward_components'],red_distance=g.distance,
                red_alpha_i_deg=np.degrees(g.alpha_i),red_alpha_j_deg=np.degrees(g.alpha_j),
                blue_alpha_i_deg=np.degrees(gb.alpha_i),blue_alpha_j_deg=np.degrees(gb.alpha_j),
                red_eligible=qualifies[0],blue_eligible=qualifies[1],
                red_dwell_at_resolution=streaks[0],blue_dwell_at_resolution=streaks[1],
                numerical_step=info['numerical_step'],events=event_rows,terminated=term,truncated=trunc,
                outcome=info['outcome'],numerical_failure=False)
            _, row['posture_samples'] = transition_samples(row,config)
            # Sampling probes deterministically does not change the actual integrator path.
            probes = []
            if inspect and (step%25==1 or any(d['continuation_substeps'] for d in info['numerical_step']) and step%5==0):
                for side, before, after, frames, controls, alive in (
                    ('Red',r0,r1,rf,info['red_controls'],lr),('Blue',b0,b1,bf,info['blue_controls'],lb)):
                    for i in np.flatnonzero(alive):
                        probes.append(dict(side=side,id=int(i),**motion_probe(before[i],after[i],frames[i],controls[i],config)))
            row['motion_probes'] = probes
            if trainer is not None:
                with torch.no_grad():
                    ci = normalize_inputs(trainer.tensor(central),trainer.config.observation_normalization)
                    ca = torch.tanh(trainer.critic.network[0](ci))
                    ai = trainer.tensor(actor['network_input'])
                    aa = torch.tanh(trainer.actor.network[0](ai))
                    row['actor_tanh_saturation_per_slot'] = (aa.abs()>.99).float().mean(-1).cpu().numpy()
                    row['critic_tanh_saturation'] = float((ca.abs()>.99).float().mean())
                    row['network_critic_input'] = ci.cpu().numpy()
            stream.write(json.dumps(serializable(row),allow_nan=False,separators=(',',':'))+'\n')
    return dict(seed=seed,scenario=scenario,policy=policy,selection='deterministic' if deterministic else 'sampled_evaluation',
                steps=step,outcome='simulation_fault' if failure else info['outcome'],failure=failure,
                red_survivors=int(env.state.friendly_alive.sum()),blue_survivors=int(env.state.adversarial_alive.sum()),
                win=bool(not failure and info['win']),events=all_events,first_steps=first,counts=counts,
                assignment_switches=switches,initial=initial,trace=str(trace),numerical_diagnostics=env.diagnostics)


def run(out, inventory):
    started = time.perf_counter()
    available = [m for m in inventory if m['status']=='ready']
    # All recorded policies evaluated on identical, explicit scenario starts.
    configs = {m['label']: m for m in available}
    if 'signed' not in configs: raise ValueError('reference signed config missing; do not silently substitute')
    models = available+[dict(label='untrained',reference='signed'),dict(label='trim',reference='signed')]
    episodes = []
    for model in models:
        label = model['label']; folder = out/label; folder.mkdir()
        if label in ('untrained','trim'):
            reference, config, _ = load_model(configs['signed']['checkpoint'])
            trainer = ProtocolMAPPOTrainer(reference.config) if label=='untrained' else None
            del reference
            provenance = dict(label=label,reference_config_from=configs['signed']['checkpoint'],
                              initialization_seed=configs['signed']['training']['seed'],
                              trained=False,environment=config.to_dict())
        else:
            trainer, config, _ = load_model(model['checkpoint'])
            provenance = model
        if trainer is not None: trainer.actor.eval(); trainer.critic.eval()
        write_json(folder/'provenance.json',provenance)
        for scenario in ('head_on','head_on_3v3'):
            for seed in range(10010000,10010020):
                before = time.perf_counter()
                ep = episode(trainer,config,seed,scenario,label,folder/f'{scenario}_{seed}.jsonl.gz')
                ep.update(model=label,elapsed_s=time.perf_counter()-before)
                episodes.append(serializable(ep))
                write_json(out/'episodes.json',episodes)
                print(f'[AUDIT] {label} {scenario} seed={seed} steps={ep["steps"]} outcome={ep["outcome"]} R={ep["red_survivors"]} B={ep["blue_survivors"]} wall={ep["elapsed_s"]:.1f}s',flush=True)
        if label not in ('untrained','trim'):
            # One separately labelled stochastic-policy rollout, not historical training actions.
            ep = episode(trainer,config,10010000,'head_on_3v3',label,folder/'sampled_head_on_3v3_10010000.jsonl.gz',deterministic=False)
            ep.update(model=label,diagnostic_only=True)
            episodes.append(serializable(ep)); write_json(out/'episodes.json',episodes)
            assert digest(model['checkpoint']) == model['sha256'], 'checkpoint modified during read-only audit'
        print(f'[MODEL_DONE] {label} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    write_json(out/'execution.json',dict(elapsed_s=time.perf_counter()-started,completed_episodes=len(episodes),
        deterministic_seeds=list(range(10010000,10010020)),torch_sample_seed='episode_seed+20000000',
        device=torch.cuda.get_device_name(0),models_unchanged=True,training_executed=False))


def physics_checks(output):
    """Bounded independent reference checks, real integrator settings unchanged."""
    from tools.validate_foundation import flight_acceptance, complex_acceptance
    from env.aircombat import AirCombatConfig
    from env.paper_dynamics import rhs as paper_rhs
    require_cuda('cuda')
    c=AirCombatConfig()
    flights=flight_acceptance();polar=complex_acceptance()
    rng=np.random.default_rng(1101);differential=[]
    corners=[(mu,z,n) for mu in (-np.pi,0,np.pi) for z in (-1,2) for n in (0,8)]
    for k in range(212):
        state=np.r_[rng.uniform(-1000,1000,3),rng.uniform(100,400),rng.uniform(-1.2,1.2),rng.uniform(-np.pi,np.pi)]
        control=np.array(corners[k]) if k<len(corners) else rng.uniform([-np.pi,-1,0],[np.pi,2,8])
        frame=initial_frame(state);d=scalar_rhs(state,control,c.gravity)
        actual=paper_rhs(state,control,gravity=c.gravity,epsilon=c.denominator_epsilon)
        fd=frame_rhs(np.r_[state[:4],frame.ravel()],control,c);du=fd[4:].reshape(3,3)[:,0]
        recovered=np.r_[fd[:4],float(du@frame[:,1]),float(du@frame[:,2])/np.cos(state[4])]
        differential.append(dict(state=state,control=control,theoretical=d,paper_rhs=actual,frame_recovered_rhs=recovered,
            paper_error=float(np.max(abs(actual-d))),frame_error=float(np.max(abs(recovered-d)))))
    for case in flights['cases']:
        state=np.array(case['initial']);control=np.array(case['control']);counters=dict(speed_clips=0,nonfinite_states=0,integration_failures=0)
        standard,ok=rk4(state,control,c,counters)
        case['instantaneous_paper_derivative']=scalar_rhs(state,control,c.gravity).tolist()
        case['angle_rk4_dt025_accepted']=ok;case['angle_rk4_dt025']=standard.tolist()
        case['frame_rk4_dt025']=case['states'][1]
        delta=np.array(case['states'][1])-standard;delta[5]=wrap_angle(delta[5])
        case['one_step_difference']=delta.tolist()
    result=dict(flights=flights,polar=polar,differential=differential,
        max_scalar_rhs_error=max(p['paper_error'] for p in differential),
        max_frame_recovered_rhs_error=max(p['frame_error'] for p in differential),
        explanation='12 corners + 200 seeded random normal states, independent scalar equations; no runtime settings changed')
    assert result['max_scalar_rhs_error']<1e-12 and result['max_frame_recovered_rhs_error']<1e-12
    # Output is new even if an earlier execution failed elsewhere.
    with Path(output).open('x',encoding='utf-8') as stream:
        json.dump(serializable(result),stream,indent=2,allow_nan=False)
    print('[PHYSICS] cases=10 polar_trajectories=10 differential=212 max_frame_rhs_error='+str(result['max_frame_recovered_rhs_error']),flush=True)


def policy_probes(source):
    """Controlled network queries only; not invented flight or weapon events."""
    out=Path(source);inventory=json.loads((out/'inventory.json').read_text())['models'];results=[]
    for model in inventory:
        if model['status']!='ready':continue
        trainer,c,_=load_model(model['checkpoint']);queries=[]
        for name,position in (('left',[1500,600,6000]),('right',[1500,-600,6000]),
                ('above',[1500,0,6600]),('below',[1500,0,5400]),('far',[5000,600,6000]),('far_right',[5000,-600,6000])):
            red=np.tile([0,0,6000,200,0,0],(3,1)).astype(float)
            blue=np.tile([*position,200,0,0],(3,1)).astype(float)
            e=AirCombatEnv(c);obs,info=e.reset(1101,dict(friendly_states=red,adversarial_states=blue,
                                                      friendly_alive=[1,0,0],adversarial_alive=[1,0,0]))
            actor=actor_step(trainer,obs,info['active_mask'],True)
            queries.append(dict(name=name,target_position=position,raw_observation=obs[0],
                network_input=actor['network_input'][0],beta_alpha=actor['beta_alpha'][0],beta_beta=actor['beta_beta'][0],
                beta_std=actor['beta_std'][0],action=actor['actual_actions'][0],physical_control=physical_actions(actor['actual_actions'][0])))
        assert digest(model['checkpoint'])==model['sha256']
        results.append(dict(model=model['label'],queries=queries))
    with (out/'policy_probes.json').open('x',encoding='utf-8') as f:json.dump(serializable(results),f,indent=2,allow_nan=False)
    print('[PROBES] '+str(len(results))+' models; 6 controlled observations each',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--inventory-only', action='store_true')
    args = p.parse_args()
    require_cuda('cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    inventory = discover()
    write_json(out/'inventory.json', dict(models=inventory, device=torch.cuda.get_device_name(0),
               torch=torch.__version__, platform=platform.platform()))
    for item in inventory:
        print(json.dumps({k: item[k] for k in ('label', 'status', 'completed_episodes', 'updates', 'sampled_steps', 'reason') if k in item}), flush=True)
    if not args.inventory_only:
        physics_checks(out/'physics_checks.json')
        policy_probes(out)
        run(out, inventory)


if __name__ == '__main__':
    main()
