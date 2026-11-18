"""Receding-horizon direct-control MPC; target center is the chosen reference."""
from dataclasses import dataclass, field
import math
import time

import numpy as np

from ..dynamics import DynamicsConfig, PointState, WingState, wrap
from ..prediction import TargetHistory, forecast
from .dynamics import AControl, BControl, direct_step


@dataclass(frozen=True)
class MPCConfig:
    planning_interval: float = .2
    a_nodes: int = 10
    a_node_dt: float = .2
    b_nodes: int = 10
    b_node_dt: float = .4
    b_max_iterations: int = 20


@dataclass
class MPCController:
    dynamics: DynamicsConfig
    history: TargetHistory
    config: MPCConfig = field(default_factory=MPCConfig)
    cached: dict = field(default_factory=dict)
    _b_solver: object = None
    _b_initial: dict = field(default_factory=dict)

    def _reference(self, target_index, target_position, node_times):
        try:
            prediction=forecast(self.history,target_index,"CV")
            return np.asarray([prediction.at(float(t)) for t in node_times]),"CV"
        except ValueError:
            return np.repeat(np.asarray(target_position,float)[None,:],len(node_times),axis=0),"OBSERVED"

    def _fallback(self,state,target):
        delta=np.asarray(target,float)-state.position
        norm=float(np.linalg.norm(delta))
        if norm<1e-9:
            delta=np.array((1.,0.,0.)); norm=1.
        direction=delta/norm
        if isinstance(state,PointState):
            wanted=direction*self.dynamics.v_max-state.velocity
            length=np.linalg.norm(wanted)
            return AControl(tuple(self.dynamics.a_max*wanted/length if length else np.zeros(3)))
        yaw=math.atan2(direction[1],direction[0])
        gamma=math.atan2(direction[2],math.hypot(direction[0],direction[1]))
        yaw_rate=float(np.clip(wrap(yaw-state.yaw)/self.config.planning_interval,
                               -self.dynamics.yaw_rate_max,self.dynamics.yaw_rate_max))
        gamma_rate=float(np.clip((gamma-state.flightpath)/self.config.planning_interval,
                                 -self.dynamics.flightpath_rate_max,self.dynamics.flightpath_rate_max))
        return self._b_control(np.array((0.,yaw_rate/self.dynamics.yaw_rate_max,
                                          gamma_rate/self.dynamics.flightpath_rate_max)))

    def _b_control(self,row):
        a=float(np.clip(row[0],-self.dynamics.braking_max,self.dynamics.acceleration_max))
        yaw=float(np.clip(row[1],-1,1))*self.dynamics.yaw_rate_max
        gamma=float(np.clip(row[2],-1,1))*self.dynamics.flightpath_rate_max
        lateral=self.dynamics.v_max*math.hypot(yaw,gamma)
        if lateral>self.dynamics.lateral_acceleration_max:
            factor=self.dynamics.lateral_acceleration_max/lateral
            yaw*=factor; gamma*=factor
        return BControl(a,yaw,gamma)

    def _solve_a(self,state,reference):
        import cvxpy as cp
        n=self.config.a_nodes; h=self.config.a_node_dt
        p=cp.Variable((n+1,3)); v=cp.Variable((n+1,3)); a=cp.Variable((n,3))
        constraints=[p[0]==state.position,v[0]==state.velocity]
        objective=0
        for k in range(n):
            constraints += [p[k+1]==p[k]+h*v[k]+.5*h*h*a[k],
                            v[k+1]==v[k]+h*a[k],
                            cp.norm(a[k],2)<=self.dynamics.a_max,
                            cp.norm(v[k+1],2)<=self.dynamics.v_max]
            objective += cp.sum_squares((p[k+1]-reference[k])/50.)
            objective += .01*cp.sum_squares(a[k]/self.dynamics.a_max)
        objective += 2*cp.sum_squares((p[n]-reference[-1])/50.)
        problem=cp.Problem(cp.Minimize(objective),constraints)
        problem.solve(solver="CLARABEL",warm_start=True,verbose=False)
        if problem.status not in ("optimal","optimal_inaccurate") or a.value is None:
            raise RuntimeError(f"A solver: {problem.status}")
        controls=[AControl(tuple(float(x) for x in row)) for row in a.value]
        return controls,dict(solver_status=problem.status,solver_iterations=problem.solver_stats.num_iters,
                             tracking_cost=float(problem.value))

    def _build_b_solver(self):
        import casadi as ca
        n=self.config.b_nodes; h=self.config.b_node_dt
        repeats=round(h/.01)
        if not math.isclose(repeats*.01,h,abs_tol=1e-10):
            raise ValueError("B node interval must be a multiple of 0.01 s")
        u=ca.SX.sym("controls",3*n)
        parameters=ca.SX.sym("state_reference",6+3*n)
        position=parameters[:3]
        speed,yaw,gamma=parameters[3],parameters[4],parameters[5]
        objective=0
        for k in range(n):
            a=u[3*k]
            yaw_rate=u[3*k+1]*self.dynamics.yaw_rate_max
            gamma_rate=u[3*k+2]*self.dynamics.flightpath_rate_max
            lateral=self.dynamics.v_max*ca.sqrt(yaw_rate*yaw_rate+gamma_rate*gamma_rate+1e-24)
            scale=ca.fmin(1,self.dynamics.lateral_acceleration_max/lateral)
            yaw_rate*=scale; gamma_rate*=scale
            for _ in range(repeats):
                next_speed=ca.fmin(self.dynamics.v_max,ca.fmax(self.dynamics.v_min,speed+a*.01))
                next_gamma=ca.fmin(self.dynamics.flightpath_max,
                                   ca.fmax(-self.dynamics.flightpath_max,gamma+gamma_rate*.01))
                v_mid=(speed+next_speed)/2
                yaw_mid=yaw+yaw_rate*.005
                gamma_mid=(gamma+next_gamma)/2
                position=position+.01*v_mid*ca.vertcat(ca.cos(gamma_mid)*ca.cos(yaw_mid),
                    ca.cos(gamma_mid)*ca.sin(yaw_mid),ca.sin(gamma_mid))
                speed=next_speed; yaw=yaw+yaw_rate*.01; gamma=next_gamma
            reference=parameters[6+3*k:9+3*k]
            error=(position-reference)/50.
            objective+=ca.dot(error,error)+.01*ca.dot(u[3*k:3*k+3],u[3*k:3*k+3])
            if k:
                difference=u[3*k:3*k+3]-u[3*(k-1):3*k]
                objective+=.02*ca.dot(difference,difference)
        objective+=2*ca.dot(error,error)
        self._b_solver=ca.nlpsol("kinematic_b_mpc","ipopt",
            {"x":u,"p":parameters,"f":objective},
            {"ipopt.print_level":0,"print_time":False,"ipopt.sb":"yes",
             "ipopt.max_iter":self.config.b_max_iterations})

    def _solve_b(self,index,state,reference):
        if self._b_solver is None:
            self._build_b_solver()
        n=self.config.b_nodes
        initial=self._b_initial.get(index,np.zeros(3*n))
        parameters=np.r_[state.position,state.speed,state.yaw,state.flightpath,
                         reference.reshape(-1)]
        lower=np.tile((-self.dynamics.braking_max,-1.,-1.),n)
        upper=np.tile((self.dynamics.acceleration_max,1.,1.),n)
        result=self._b_solver(x0=initial,p=parameters,lbx=lower,ubx=upper)
        status=self._b_solver.stats()["return_status"]
        if not self._b_solver.stats()["success"] and status != "Maximum_Iterations_Exceeded":
            raise RuntimeError(f"B solver: {status}")
        solution=np.asarray(result["x"]).reshape(n,3)
        if not np.isfinite(solution).all() or not math.isfinite(float(result["f"])):
            raise RuntimeError("B solver returned nonfinite plan")
        self._b_initial[index]=np.vstack((solution[1:],solution[-1])).reshape(-1)
        controls=[self._b_control(row) for row in solution]
        return controls,dict(solver_status="optimal" if self._b_solver.stats()["success"]
                             else "iteration_limit",
            solver_iterations=int(self._b_solver.stats()["iter_count"]),
            tracking_cost=float(result["f"]))

    def command(self,index,state,target_index,target_position,time_now):
        cached=self.cached.get(index)
        if cached is not None and cached[0]==target_index and time_now-cached[1]<self.config.planning_interval-1e-9:
            return cached[2],dict(status="HELD_MPC",solver_status="held")
        if cached is not None and cached[0]!=target_index:
            self._b_initial.pop(index,None)
        model="A" if isinstance(state,PointState) else "B"
        nodes=self.config.a_nodes if model=="A" else self.config.b_nodes
        node_dt=self.config.a_node_dt if model=="A" else self.config.b_node_dt
        reference,source=self._reference(target_index,target_position,
                                           node_dt*np.arange(1,nodes+1))
        start=time.perf_counter()
        try:
            controls,diagnostics=(self._solve_a(state,reference) if model=="A"
                                  else self._solve_b(index,state,reference))
            first=controls[0]
            # Reject any plan whose first action cannot be executed by the exact model.
            direct_step(state,first,self.dynamics,.01)
            fallback=False
        except (RuntimeError,ValueError,ArithmeticError) as error:
            first=self._fallback(state,target_position)
            diagnostics=dict(solver_status=f"failed:{type(error).__name__}",
                             solver_iterations=0,tracking_cost=None)
            fallback=True
        elapsed=time.perf_counter()-start
        self.cached[index]=(target_index,time_now,first)
        return first,dict(status="FALLBACK" if fallback else "MPC_READY",
            model_type=model,target_id=target_index,reference_source=source,
            solve_time_s=elapsed,fallback_used=fallback,**diagnostics)
