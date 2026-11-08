import numpy as np

from algorithm.interception import episode
from env.interception import InterceptionEnv
from tools.interception_high_speed_checks import config, contact_checks, motion_checks


def test_high_speed_patrol_initialization_and_matching_s1_s2():
    for scenario in ("fixed", "random"):
        a, ia = InterceptionEnv(config(task="S1")).reset(1200, {"scenario": scenario})
        b, ib = InterceptionEnv(config(task="S2")).reset(1200, {"scenario": scenario})
        np.testing.assert_array_equal(a, b)
        p, e = ia["pursuers"][:3], ia["evaders"][0]
        np.testing.assert_array_equal(ia["pursuers"], ib["pursuers"])
        np.testing.assert_array_equal(ia["evaders"], ib["evaders"])
        assert np.all((15 <= np.linalg.norm(p[:,3:], axis=1)) & (np.linalg.norm(p[:,3:], axis=1) <= 20))
        assert 18 <= np.linalg.norm(e[3:]) <= 23
        assert np.all(np.sum(p[:,:2]*p[:,3:5], axis=1) < 0)
        np.testing.assert_allclose(a[0,:3], p[0,:3]/650)
        np.testing.assert_allclose(a[0,3:6], p[0,3:]/30)
        assert a.shape == (5,52) and ia["centralized_state"].shape == (54,)


def test_high_speed_velocity_limits_and_analytic_contact_cases():
    assert len(motion_checks()) == 8
    cases = contact_checks()
    assert len(cases) == 13
    assert max(abs(row["time_error"]) for row in cases if row["time_error"] is not None) <= 1e-7


def test_high_speed_read_only_audit_leaves_episode_identical():
    c = config(task="S2")
    normal, a = episode(c, 1200, {"scenario":"random"}, "predictive", trace=True)
    audited, b = episode(c, 1200, {"scenario":"random"}, "predictive", trace=True, audit=True)
    assert normal == {k: audited[k] for k in normal}
    np.testing.assert_equal(a, b)
