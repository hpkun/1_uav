"""Render the actual 15v15 seed geometry from an episode summary."""
import argparse
import json
from pathlib import Path


def report(summary):
    point=lambda values:"("+", ".join(f"{value:.9f}" for value in values)+")"
    lines=["# V3.5 15v15 exact scene", "", "Scenario ID: "+summary["scenario_id"],
           "",f"Assignment cost: {summary['assignment_total_cost_m']:.12f} m", "",
           "## Defender positions", "",
           "| Defender | Position (m) | Assigned intruder |",
           "| --- | --- | --- |"]
    for i,start in enumerate(summary["defender_starts"]):
        assignment=summary["original_assignment"]
        target=assignment[str(i)] if str(i) in assignment else assignment[i]
        lines.append(f"| D{i} | {point(start)} | I{target} |")
    lines += ["", "## Intruder paths", "",
              "| Intruder | Jittered P0 (m) | P1 (m) | P2 (m) | P3 (m) | Region entry (s) | SHA-256 |",
              "| --- | --- | --- | --- | --- | ---: | --- |"]
    for i,path in enumerate(summary["intruder_paths"]):
        p1,p2=path["control_points"]
        lines.append(f"| I{i} | {point(path['start'])} | {point(p1)} | "
            f"{point(p2)} | {point(path['goal'])} | "
            f"{path['protected_region_entry_time_s']:.9f} | {path['path_sha256']} |")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args(argv)
    content=report(json.loads(args.summary.read_text(encoding="utf-8")))
    with args.output.open("x",encoding="utf-8") as stream:
        stream.write(content)
    print(args.output)


if __name__=="__main__":
    main()
