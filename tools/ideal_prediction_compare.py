"""Compare paired Zuo linear and Tan quadratic forecasts on blue observations."""
import argparse
import csv
import json
from pathlib import Path

from prediction.comparison import METHODS,compare,read_observations

PREDICTION_FIELDS=("scenario_id","seed","target_index","observation_time","method",
                   "horizon_steps","predicted_time","predicted_x","predicted_y","predicted_z",
                   "actual_x","actual_y","actual_z","error_m","valid",
                   "predicted_displacement_m")
TABLE_FIELDS=("scenario_id","target_index","method","ValidSamples_1","ValidSamples_2",
              "MAE_1","MAE_2","RMSE_1","RMSE_2","MaxError_1","MaxError_2")


def run(observations,out,plot=False):
    result=compare(read_observations(observations))
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"predictions.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(result["predictions"])
    metrics={key:value for key,value in result.items() if key!="predictions"}
    with (out/"metrics.json").open("x",encoding="utf-8") as stream:
        json.dump(metrics,stream,indent=2,allow_nan=False)
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
    args=parser.parse_args(argv)
    result=run(args.observations,args.out,args.plot)
    for target,summary in result["targets"].items():
        for method in METHODS:
            values=summary["methods"][method]
            print(f"{target} {method}: n1={values['ValidSamples_1']} MAE1={values['MAE_1']} m | "
                  f"n2={values['ValidSamples_2']} MAE2={values['MAE_2']} m")


if __name__=="__main__":
    main()
