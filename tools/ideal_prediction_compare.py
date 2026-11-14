"""Compare paired Zuo linear and Tan quadratic forecasts on blue observations."""
import argparse
import csv
import json
from pathlib import Path

from prediction.comparison import METHODS,compare,compare_horizons,compare_positions,read_observations

PREDICTION_FIELDS=("scenario_id","seed","target_index","observation_time","method",
                   "horizon_steps","predicted_time","predicted_x","predicted_y","predicted_z",
                   "actual_x","actual_y","actual_z","error_m","valid",
                   "predicted_displacement_m")
TABLE_FIELDS=("scenario_id","target_index","method","ValidSamples_1","ValidSamples_2",
              "MAE_1","MAE_2","RMSE_1","RMSE_2","MaxError_1","MaxError_2")


def run_extended(observations,out,horizons=(1,5,10,20),plot_horizon=10,tan_limit=None):
    """Write a compact paired, multi-horizon report from one saved observation file."""
    data=read_observations(observations)
    result=compare_positions(data,horizons,tan_limit)
    if plot_horizon not in result["horizon_steps"]:
        raise ValueError("plot_horizon must be one of the compared horizons")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    fields=list(result["rows"][0]) if result["rows"] else []
    with (out/"comparison.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    summary={key:value for key,value in result.items() if key!="rows"}
    with (out/"summary.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    target_count=len(data["targets"])
    fig=plt.figure(figsize=(13,5*target_count))
    error_fig,error_axes=plt.subplots(target_count,1,figsize=(10,3.5*target_count),squeeze=False)
    for index,target in enumerate(data["targets"]):
        j=target["target_index"]
        truth=np.asarray([row["position"] for row in target["observations"]],float)
        selected=[row for row in result["rows"] if row["target_index"]==j and row["horizon_steps"]==plot_horizon]
        samples=[selected[i] for i in np.linspace(0,len(selected)-1,min(12,len(selected)),dtype=int)] if selected else []
        ax3=fig.add_subplot(target_count,2,2*index+1,projection="3d")
        ax2=fig.add_subplot(target_count,2,2*index+2)
        for axis,dimensions in ((ax3,(0,1,2)),(ax2,(0,1))):
            if len(dimensions)==3:
                axis.plot(truth[:,0],truth[:,1],truth[:,2],label="Reference truth",color="black",lw=1)
            else:
                axis.plot(truth[:,0],truth[:,1],label="Reference truth",color="black",lw=1)
            for prefix,label,marker in (("zuo","Zuo","x"),("three_point","Three point","+")):
                points=np.asarray([[row[f"{prefix}_{a}"] for a in "xyz"] for row in samples])
                if len(points):
                    if len(dimensions)==3:
                        axis.scatter(points[:,0],points[:,1],points[:,2],label=label,marker=marker,s=30)
                    else:
                        axis.scatter(points[:,0],points[:,1],label=label,marker=marker,s=30)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            if len(dimensions)==3: axis.set_zlabel("z (m)")
            axis.set_title(f"I{j}: +{plot_horizon*data['dt']:.2f} s")
            axis.legend(fontsize=8)
        error_axis=error_axes[index,0]
        error_axis.plot([row["observation_time"] for row in selected],
                        [row["zuo_error_m"] for row in selected],label="Zuo")
        error_axis.plot([row["observation_time"] for row in selected],
                        [row["three_point_error_m"] for row in selected],label="Three point")
        limited=[row for row in selected if row["tan_horizon_status"]=="HORIZON_LIMITED"]
        if limited:
            error_axis.scatter([row["observation_time"] for row in limited],
                               [row["three_point_error_m"] for row in limited],
                               s=8,color="gray",label="Beyond exploratory Tan cap",zorder=3)
        error_axis.set(title=f"I{j}: +{plot_horizon*data['dt']:.2f} s",xlabel="Observation time (s)",
                       ylabel="3D position error (m)")
        error_axis.legend()
    fig.tight_layout()
    fig.savefig(out/"trajectory_overlay.png",dpi=150)
    plt.close(fig)
    error_fig.tight_layout()
    error_fig.savefig(out/"error_curve.png",dpi=150)
    plt.close(error_fig)
    return result


def run(observations,out,plot=False,horizons=None):
    data=read_observations(observations)
    result=compare(data)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"predictions.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(result["predictions"])
    metrics={key:value for key,value in result.items() if key!="predictions"}
    with (out/"metrics.json").open("x",encoding="utf-8") as stream:
        json.dump(metrics,stream,indent=2,allow_nan=False)
    if horizons is not None:
        with (out/"horizons.json").open("x",encoding="utf-8") as stream:
            json.dump(compare_horizons(data,horizons),stream,indent=2,allow_nan=False)
    table=[]
    for target,summary in result["targets"].items():
        for method in METHODS:
            values=summary["methods"][method]
            table.append(dict(scenario_id=result["scenario_id"],target_index=int(target[1:]),
                method=method,**{key:values[key] for key in TABLE_FIELDS[3:]}))
    with (out/"comparison.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(table)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        labels=[f"{target} +{h}" for target in result["targets"] for h in (1,2)]
        x=np.arange(len(labels))
        fig,ax=plt.subplots(figsize=(max(7,len(labels)*1.6),4))
        for offset,method in ((-.18,"zuo_linear"),(.18,"tan_quadratic")):
            values=[result["targets"][target]["methods"][method][f"MAE_{h}"] or 0
                    for target in result["targets"] for h in (1,2)]
            ax.bar(x+offset,values,width=.36,label=method)
        ax.set(xticks=x,xticklabels=labels,ylabel="Mean 3D position error (m)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/"comparison.png",dpi=150)
        plt.close(fig)
        fig,axes=plt.subplots(len(result["targets"]),1,figsize=(9,3.5*len(result["targets"])),squeeze=False)
        for axis,(target,_) in zip(axes[:,0],result["targets"].items()):
            j=int(target[1:])
            for method in METHODS:
                for horizon in (1,2):
                    rows=[row for row in result["predictions"] if row["target_index"]==j
                          and row["method"]==method and row["horizon_steps"]==horizon
                          and row["valid"]]
                    axis.plot([row["observation_time"] for row in rows],
                              [row["error_m"] for row in rows],label=f"{method} +{horizon}")
            axis.set(title=target,xlabel="Observation time (s)",ylabel="3D position error (m)")
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out/"error_curve.png",dpi=150)
        plt.close(fig)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--horizons",type=int,nargs="+",help="optional grid horizons, e.g. 1 2 5 10")
    parser.add_argument("--extended",action="store_true",help="compact paired position report")
    parser.add_argument("--plot-horizon",type=int,default=10)
    parser.add_argument("--tan-eta",type=float,help="enable exploratory Tan horizon cap")
    parser.add_argument("--tan-min",type=float,default=.1)
    parser.add_argument("--tan-max",type=float,default=2.)
    parser.add_argument("--tan-epsilon-acceleration",type=float,default=1e-6)
    args=parser.parse_args(argv)
    if args.extended:
        limit=(dict(eta=args.tan_eta,minimum=args.tan_min,maximum=args.tan_max,
                    epsilon_acceleration=args.tan_epsilon_acceleration)
               if args.tan_eta is not None else None)
        result=run_extended(args.observations,args.out,args.horizons or (1,5,10,20),
                            args.plot_horizon,limit)
        print(f"paired={len(result['rows'])} | horizons={result['horizon_steps']} | targets={len(result['targets'])}")
        return
    if args.tan_eta is not None:
        parser.error("--tan-eta requires --extended")
    result=run(args.observations,args.out,args.plot,args.horizons)
    for target,summary in result["targets"].items():
        for method in METHODS:
            values=summary["methods"][method]
            print(f"{target} {method}: n1={values['ValidSamples_1']} MAE1={values['MAE_1']} m | "
                  f"n2={values['ValidSamples_2']} MAE2={values['MAE_2']} m")


if __name__=="__main__":
    main()
