"""Report physical statistics without tuning reproduction assumptions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from uav_combat.environment import PaperUAVCombatEnv
from uav_combat.environment.geometry import PaperAirCombatGeometry
from uav_combat.models import AircraftState


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); p.add_argument("--episodes",type=int); p.add_argument("--output",default="outputs/environment_statistics.json"); args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    episodes=args.episodes or (10 if args.smoke else 100); mc=200 if args.smoke else 2000; env=PaperUAVCombatEnv(root/"configs/paper_environment.yaml")
    reset_rows=[]
    for seed in range(episodes):
        env.reset(seed); rc=np.mean([[s.x,s.y] for s in env.red],0); bc=np.mean([[s.x,s.y] for s in env.blue],0)
        spacing=min(np.linalg.norm(env.red[i].as_array()[:3]-env.red[j].as_array()[:3]) for i in range(4) for j in range(i))
        reset_rows.append((np.linalg.norm(rc-bc),spacing,np.mean([s.altitude for s in env.red+env.blue]),np.mean([s.v for s in env.red+env.blue])))
    outcomes=[]; contacts=[]; finite=True
    for seed in range(episodes):
        obs,_=env.reset(1000+seed); first=None
        while True:
            if first is None:
                distance=min(np.linalg.norm(r.as_array()[:3]-b.as_array()[:3]) for r in env.red if r.alive for b in env.blue if b.alive)
                if distance<=4000: first=env.steps*env.dt
            red_actions=np.stack([env.fixed_policy.action(s,env.blue)[0] for s in env.red]); obs,_,term,trunc,info=env.step(red_actions); finite=finite and np.isfinite(obs).all()
            if first is None:
                distance=min(np.linalg.norm(r.as_array()[:3]-b.as_array()[:3]) for r in env.red for b in env.blue)
                if distance<=4000: first=env.steps*env.dt
            if term or trunc: outcomes.append(info); contacts.append(first); break
    sweep={}
    for distance in (500,1000,2000,3000,4000):
        for ata in (0,5,15,30):
            for ha in (0,5,15,30):
                g=PaperAirCombatGeometry(distance,np.deg2rad(ata),0,np.deg2rad(ha),0); rng=np.random.default_rng(distance+ata*100+ha)
                sweep[f"d{distance}_ata{ata}_ha{ha}"]=float(np.mean([env.weapon.sample_hit(g,rng) for _ in range(mc)]))
    controller=[]
    for speed in (150,225,300):
        s=AircraftState(0,0,-3000,speed,0,0); rates=[]; saturations=0
        for _ in range(20):
            target,control=env.controller.control_from_action(s,np.array([1,1,1],float),env.spec); derivative=env.dynamics.derivatives(s,control); rates.append((derivative[5],derivative[4],derivative[3],control.nx,control.nz,control.phi)); saturations+=int(abs(control.nx) in (abs(env.spec.nx_min),abs(env.spec.nx_max)) or abs(control.nz) in (abs(env.spec.nz_min),abs(env.spec.nz_max)) or abs(control.phi)>=env.spec.phi_max-1e-8); s=env.integrator.step(s,control,env.dynamics,env.spec)
        array=np.asarray(rates); controller.append({"initial_speed":speed,"mean_realized_yaw_rate":float(array[:,0].mean()),"mean_pitch_rate":float(array[:,1].mean()),"mean_speed_acceleration":float(array[:,2].mean()),"mean_nx":float(array[:,3].mean()),"mean_nz":float(array[:,4].mean()),"mean_phi":float(array[:,5].mean()),"saturation_fraction":saturations/20,"final_speed":s.v,"finite":bool(np.isfinite(array).all())})
    mean=lambda key:float(np.mean([r[key] for r in outcomes]))
    rr=np.asarray(reset_rows)
    result={"episodes":episodes,"random_reset":{"initial_inter_team_center_distance_mean":float(rr[:,0].mean()),"min_friendly_spacing_mean":float(rr[:,1].mean()),"initial_altitude_mean":float(rr[:,2].mean()),"initial_speed_mean":float(rr[:,3].mean())},"rule_vs_rule":{"red_win_rate":mean("red_win"),"blue_win_rate":mean("blue_win"),"draw_timeout_rate":mean("draw_or_timeout"),"mean_episode_length":mean("episode_length"),"red_attack_kills":mean("red_attack_kills"),"blue_attack_kills":mean("blue_attack_kills"),"red_boundary_deaths":mean("red_boundary_losses"),"blue_boundary_deaths":mean("blue_boundary_losses")},"weapon_hit_probability":sweep,"controller":controller,"time_to_contact_seconds_mean":float(np.mean([x for x in contacts if x is not None])) if any(x is not None for x in contacts) else None,"finite_state_checks":bool(finite and all(row["finite"] for row in controller))}
    path=root/args.output; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
