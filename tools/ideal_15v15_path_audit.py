"""Audit the frozen 15v15 Level-2 geometry without running guidance."""
import argparse
import json
from pathlib import Path

import numpy as np

from env.ideal_interception.config import formal_15v15_level2_config,PROTOCOL
from env.ideal_interception.scenario import initialize,diagnostics


def audit(seeds=range(100)):
    config=formal_15v15_level2_config()
    hashes=set()
    rows=[]
    invalid=outside=late=duplicate=0
    for seed in seeds:
        defenders,intruders,paths,assignment,_,scenario_id=initialize(config,seed)
        report=diagnostics(config,paths)
        for index,(path,geometry) in enumerate(zip(paths,report["paths"])):
            spec=path.specification()
            digest=spec["polyline_sha256"]
            finite=bool(np.isfinite(path.points).all() and np.isfinite(path.lengths).all())
            nondegenerate=bool(np.all(path.lengths>0))
            speeds=[float(np.linalg.norm(path.sample(t)[1])) for t in
                    (0.,path.duration*.25,path.duration*.5,path.duration*.75)]
            constant_speed=bool(np.allclose(speeds,config.intruder_speeds[index],rtol=1e-12,atol=1e-10))
            valid=finite and nondegenerate and constant_speed and len(digest)==64
            invalid+=not valid
            outside+=not geometry["within_bounds"]
            late+=not geometry["entry_within_episode"]
            duplicate+=digest in hashes
            hashes.add(digest)
            rows.append(dict(seed=seed,intruder=index,path_sha256=digest,
                protected_region_entry_time_s=geometry["protected_region_entry_time"],
                within_bounds=geometry["within_bounds"],valid=valid,
                scenario_id=scenario_id))
    return dict(protocol=PROTOCOL,scale="15v15",seeds=list(seeds),path_count=len(rows),
        invalid_count=invalid,out_of_bounds_count=outside,late_entry_count=late,
        duplicate_hash_count=duplicate,paths=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    result=audit()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print({key:result[key] for key in ("path_count","invalid_count",
        "out_of_bounds_count","late_entry_count","duplicate_hash_count")})


if __name__=="__main__":
    main()
