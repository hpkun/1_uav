"""Nominal 20v20 layout, duplicate-permitting nearest assignment, and audit outputs."""
import csv
import json
import math

import numpy as np
import pytest

from env.ideal_interception.assignment import nearest_distance_assignment
from env.ideal_interception.config import (PROTOCOL, DEFENDER_SLOTS, INTRUDER_SLOTS,
    OBS_DIM, STATE_DIM, FIFTEEN_DEFENDER_CIRCLE, FIFTEEN_INTRUDER_WEDGE,
    SIX_INTRUDER_WEDGE)
from env.ideal_interception.large_scale_20v20 import (TWENTY_COUNT, DEFENDER_CENTER,
    TWENTY_DEFENDER_RADIUS_M, NEW_INTRUDER_ANCHORS, initial_positions,
    build_audit, segments_cross_interior, monotonic_y_inversions)
from tools.ideal_20v20_assignment_audit import main


def test_nominal_positions_and_frozen_15v15_anchors():
    defenders, intruders = initial_positions()
    assert defenders.shape == intruders.shape == (20, 3)
    assert np.isfinite(defenders).all() and np.isfinite(intruders).all()
    for positions in (defenders, intruders):
        assert (positions >= (-300., -300., 0.)).all()
        assert (positions <= (300., 300., 120.)).all()
    assert DEFENDER_CENTER == (-120., 0., 60.)
    assert TWENTY_DEFENDER_RADIUS_M == pytest.approx(60.*math.sqrt(20/15))
    np.testing.assert_allclose(np.linalg.norm(defenders[:, :2]-DEFENDER_CENTER[:2], axis=1),
                               TWENTY_DEFENDER_RADIUS_M, rtol=0, atol=1e-12)
    assert np.all(defenders[:, 2] == 60.)
    np.testing.assert_array_equal(intruders[:15], FIFTEEN_INTRUDER_WEDGE)
    np.testing.assert_array_equal(intruders[15:], NEW_INTRUDER_ANCHORS)
    assert PROTOCOL == "uav-ideal-interception-v3.5.0"
    assert (DEFENDER_SLOTS, INTRUDER_SLOTS, OBS_DIM, STATE_DIM) == (15, 15, 265, 444)
    assert FIFTEEN_DEFENDER_CIRCLE == tuple(
        (-120.+60.*math.cos(2.*math.pi*i/15),
         60.*math.sin(2.*math.pi*i/15), 60.) for i in range(15))
    assert FIFTEEN_INTRUDER_WEDGE == SIX_INTRUDER_WEDGE + (
        (250.,-90.,60.), (250.,-30.,60.), (250.,30.,60.), (250.,90.,60.),
        (270.,-120.,60.), (270.,-60.,60.), (270.,0.,60.), (270.,60.,60.),
        (270.,120.,60.))


def test_nearest_assignment_allows_duplicate_targets_and_3d_distances():
    summary = build_audit()
    defenders = np.asarray(summary["defender_positions"])
    intruders = np.asarray(summary["intruder_positions"])
    assignment = {int(i): j for i, j in summary["assignment"].items()}
    assert summary["defender_count"] == summary["intruder_count"] == TWENTY_COUNT
    assert len(summary["pairs"]) == 20
    assert set(assignment) == set(range(20))
    assert set(assignment.values()) == {0}
    assert assignment == nearest_distance_assignment(defenders, intruders,
        np.ones(20, dtype=bool), np.ones(20, dtype=bool))
    distances = []
    for i, pair in enumerate(summary["pairs"]):
        assert pair["defender"] == i and pair["intruder"] == assignment[i]
        distance = float(np.linalg.norm(defenders[i]-intruders[assignment[i]]))
        assert math.isfinite(distance) and distance > 0
        assert pair["distance_m"] == pytest.approx(distance)
        distances.append(distance)
    stats = summary["statistics"]
    assert stats["total_distance_m"] == pytest.approx(sum(distances))
    assert stats["mean_distance_m"] == pytest.approx(np.mean(distances))
    assert stats["std_distance_m"] == pytest.approx(np.std(distances))
    assert stats["minimum_distance_m"] == pytest.approx(min(distances))
    assert stats["maximum_distance_m"] == pytest.approx(max(distances))
    assert stats["assignment_line_crossings_xy"] == 0
    assert stats["monotonic_y_inversions"] == 0
    json.dumps(summary, allow_nan=False)


def test_crossing_excludes_shared_endpoints_and_y_inversion_counts():
    assert segments_cross_interior((0,0),(2,2),(0,2),(2,0))
    assert not segments_cross_interior((0,0),(1,1),(1,1),(2,0))
    assert not segments_cross_interior((0,0),(190,0),(0,1),(190,0))
    assert not segments_cross_interior((0,0),(2,0),(1,0),(3,0))
    defenders = np.column_stack((np.zeros(20), np.arange(20), np.zeros(20)))
    intruders = defenders.copy()
    assignment = {i: i for i in range(20)}
    assert monotonic_y_inversions(defenders, intruders, assignment) == 0
    assignment[0], assignment[1] = 1, 0
    assert monotonic_y_inversions(defenders, intruders, assignment) == 1


def test_cli_outputs_are_write_once(tmp_path):
    out = tmp_path/"audit"
    main(["--out", str(out)])
    assert {path.name for path in out.iterdir()} == {
        "summary.json", "assignment.csv", "assignment_xy.png", "distance_matrix.png"}
    saved = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    assert saved == build_audit()
    with (out/"assignment.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20 and [int(row["defender"]) for row in rows] == list(range(20))
    assert all(path.stat().st_size > 10_000 for path in
               (out/"assignment_xy.png", out/"distance_matrix.png"))
    with pytest.raises(SystemExit):
        main(["--out", str(out)])
