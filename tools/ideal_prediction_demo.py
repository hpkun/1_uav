"""Offline three-point intruder prediction and error export from an ideal trace."""
import argparse
import csv
import json
from pathlib import Path

from prediction.evaluation import evaluate_trace
from tools.ideal_interception_timeline import Timeline

CSV_FIELDS=("scenario_id","seed","target_index","observation_time","horizon_steps",
            "predicted_time","predicted_x","predicted_y","predicted_z",
            "actual_x","actual_y","actual_z","error_m","valid")


def run(trace,out,plot=False,threshold=None):
    result=evaluate_trace(Timeline(trace),threshold)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for name,value in (("predictions.json",result),
                       ("metrics.json",dict(scenario_id=result["scenario_id"],seed=result["seed"],
                                            method=result["method"],metrics=result["metrics"],
                                            accuracy_threshold_m=threshold))):
        with (out/name).open("x",encoding="utf-8") as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
    with (out/"predictions.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["predictions"])
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure,axis=plt.subplots(figsize=(9,4))
        for j in range(len(result["metrics"]["targets"])):
            for horizon in (1,2):
                rows=[row for row in result["predictions"] if row["target_index"]==j
                      and row["horizon_steps"]==horizon and row["valid"]]
                axis.plot([row["observation_time"] for row in rows],
                          [row["error_m"] for row in rows],label=f"I{j} +{horizon} step")
        axis.set(xlabel="Observation time (s)",ylabel="3D position error (m)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(out/"error_curve.png",dpi=150)
        plt.close(figure)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--threshold",type=float)
    args=parser.parse_args(argv)
    result=run(args.trace,args.out,args.plot,args.threshold)
    for target,metrics in result["metrics"]["targets"].items():
        print(f"{target}: n1={metrics['ValidSamples_1']} mae1={metrics['MAE_1']} m | "
              f"n2={metrics['ValidSamples_2']} mae2={metrics['MAE_2']} m")


if __name__=="__main__":
    main()
