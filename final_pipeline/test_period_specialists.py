"""Focused tests for the release-period specialist option in train.py.

Run from final_pipeline/:  python -m unittest test_period_specialists -v
"""
import os, sys, unittest
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T


def synthetic_frame(n=240, seed=0):
    """Small train-like frame with duplicate uris, both periods, and a few unparseable dates."""
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({c: rng.random(n) for c in T.AUDIO})
    base["duration_ms"] = rng.integers(90_000, 300_000, n).astype(float)
    base["tempo"] = rng.uniform(60, 180, n); base["loudness"] = rng.uniform(-30, 0, n)
    base["key"] = rng.integers(0, 12, n).astype(float); base["mode"] = rng.integers(0, 2, n).astype(float)
    base["time_signature"] = 4.0
    years = rng.choice([1985, 2005, 2019, 2020, 2021, 2024], n)
    base["track_album_release_date"] = [f"{yr}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}" for yr in years]
    base.loc[:2, "track_album_release_date"] = ["", "bad", "nan"]            # unknown-period rows
    base["uri"] = [f"spotify:track:{i // 2:05d}" for i in range(n)]       # every uri appears twice
    base[T.TARGET] = np.where(rng.random(n) < 0.4, "High", "Low")
    return base


class BoundaryAndRouting(unittest.TestCase):
    def test_2020_boundary_and_precisions(self):
        periods = T.release_periods(["2019-12-31", "2020-01-01", "2020", "1998-03", "2024-11-22"])
        np.testing.assert_array_equal(periods, [0, 1, 1, 0, 1])

    def test_missing_or_invalid_dates_fall_back_to_global(self):
        periods = T.release_periods([None, "", "bad", "nan", "20-01-01"])
        np.testing.assert_array_equal(periods, [-1, -1, -1, -1, -1])
        routed = T.route_predictions(np.full(5, 0.4), np.full(5, 0.2), np.full(5, 0.8), periods)
        np.testing.assert_array_equal(routed, np.full(5, 0.4))

    def test_routing_picks_matching_specialist_row_by_row(self):
        periods = np.array([0, 1, -1, 1, 0])
        g, o, r = np.array([.1, .2, .3, .4, .5]), np.array([.11, .22, .33, .44, .55]), np.array([.91, .92, .93, .94, .95])
        np.testing.assert_array_equal(T.route_predictions(g, o, r, periods), [.11, .92, .3, .94, .55])

    def test_custom_cut(self):
        np.testing.assert_array_equal(T.release_periods(["2017", "2018-01-01"], cut=2018), [0, 1])


class GroupSeparation(unittest.TestCase):
    def test_specialist_training_rows_are_in_period_and_never_share_uri_with_validation(self):
        tr = synthetic_frame()
        y = (tr[T.TARGET] == "High").astype(int).values
        period = T.release_periods(tr["track_album_release_date"])
        folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0).split(tr, y, groups=tr["uri"]))
        for r in (0, 1):
            for (rtri, vai), (tri, vai0) in zip(T.regional_folds(folds, period, r), folds):
                self.assertTrue((period[rtri] == r).all())
                np.testing.assert_array_equal(vai, vai0)                     # validation part untouched
                self.assertFalse(set(tr["uri"].iloc[rtri]) & set(tr["uri"].iloc[vai]))
                self.assertTrue(set(rtri) <= set(tri))                       # subset of the outer training part
        for tri, vai in folds:                                               # both copies of a uri stay together
            self.assertFalse(set(tr["uri"].iloc[tri]) & set(tr["uri"].iloc[vai]))

    def test_specialist_label_features_use_only_period_rows(self):
        """Flipping labels of out-of-period training rows must not change the specialist's feature matrices."""
        tr = synthetic_frame(); te = synthetic_frame(n=40, seed=1)
        y = (tr[T.TARGET] == "High").astype(int).values
        period = T.release_periods(tr["track_album_release_date"])
        X, Xte = T.build_static_features(tr, te, "v2")
        folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0).split(tr, y, groups=tr["uri"]))
        rf = T.regional_folds(folds, period, 1)
        y_flipped = y.copy(); y_flipped[period != 1] = 1 - y_flipped[period != 1]
        a = T.fold_matrices(X, Xte, tr, te, y, rf, seed=0)[0]
        b = T.fold_matrices(X, Xte, tr, te, y_flipped, rf, seed=0)[0]
        for m1, m2 in zip(a, b):
            pd.testing.assert_frame_equal(m1, m2)


class OutputAlignment(unittest.TestCase):
    def test_submission_keeps_test_order_and_ids(self):
        te = synthetic_frame(n=30, seed=2); te["ID"] = np.arange(100, 130)[::-1]   # deliberately non-sorted ids
        ss = pd.DataFrame({"ID": te["ID"].values, T.TARGET: 0.5})
        pred = np.linspace(0, 1, 30)
        sub = T.build_submission(ss, te["ID"].values, pred)
        self.assertEqual(list(sub.columns), ["ID", T.TARGET])
        np.testing.assert_array_equal(sub["ID"].values, te["ID"].values)
        np.testing.assert_array_equal(sub[T.TARGET].values, pred)

    def test_mix_is_elementwise_average_of_global_and_routed(self):
        periods = np.array([0, 1, -1]); g, o, r = np.array([.2, .4, .6]), np.array([.0, .0, .0]), np.array([1., 1., 1.])
        routed = T.route_predictions(g, o, r, periods)
        np.testing.assert_allclose(0.5 * g + 0.5 * routed, [.1, .7, .6])


if __name__ == "__main__":
    unittest.main()
