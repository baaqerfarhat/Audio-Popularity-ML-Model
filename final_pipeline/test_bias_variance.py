"""Checks for group-safe sampling and label-isolated learning-curve features."""
import unittest
import numpy as np
import pandas as pd

from analyze_bias_variance import metrics, nested_training_subsets
from train import AUDIO, build_static_features, fold_matrices


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.tr = pd.DataFrame({c: rng.uniform(.05, .95, 80) for c in AUDIO})
        self.tr["duration_ms"] *= 200000
        self.tr["uri"] = np.repeat(np.arange(40), 2)
        self.tr["track_album_release_date"] = np.tile(["2000-01-01", "2001-01-01"], 40)
        self.tr["key"], self.tr["mode"], self.tr["time_signature"] = 0, 1, 4
        self.y = np.repeat(np.arange(40) % 2, 2)
        self.te = self.tr.iloc[:4].copy()
        self.te["uri"] += 100
        self.X, self.Xt = build_static_features(self.tr, self.te)

    def test_subsets_nested_and_do_not_split_tracks(self):
        subsets = nested_training_subsets(np.arange(60), self.X, self.y, self.tr.uri.to_numpy(), 0)
        self.assertTrue(set(subsets[.25]) < set(subsets[.5]) < set(subsets[1.]))
        for rows in subsets.values():
            self.assertTrue(self.tr.iloc[rows].groupby("uri").size().eq(2).all())
            self.assertEqual(len(np.unique(self.y[rows])), 2)

    def test_excluded_labels_cannot_change_learning_curve_features(self):
        folds = [(np.arange(40), np.arange(60, 80))]
        original = fold_matrices(self.X, self.Xt, self.tr, self.te, self.y, folds, 0, False)[0]
        altered = self.y.copy()
        altered[40:] = 1 - altered[40:]
        changed = fold_matrices(self.X, self.Xt, self.tr, self.te, altered, folds, 0, False)[0]
        for left, right in zip(original, changed):
            pd.testing.assert_frame_equal(left, right)

    def test_metrics_and_invalid_probabilities(self):
        result = metrics(np.array([0, 1]), np.array([.1, .9]), np.ones(2))
        self.assertEqual(result["auc"], 1.)
        self.assertAlmostEqual(result["brier"], .01)
        self.assertAlmostEqual(result["logloss"], -np.log(.9))
        with self.assertRaises(ValueError):
            metrics(np.array([0, 1]), np.array([np.nan, .9]), np.ones(2))


if __name__ == "__main__":
    unittest.main()
