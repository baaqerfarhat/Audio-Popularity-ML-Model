"""Check the distance geometry and preserve the original audio-only baseline."""
import unittest

import numpy as np
import pandas as pd

from train import KNN_SPACE, knn_coordinates, knn_date


class KnnCoordinatesTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({name: [0.2, 0.2, 0.2, 0.8] for name in KNN_SPACE})
        self.rows["duration_ms"] = [180000, 180000, 180000, 240000]
        self.rows["key"] = [0, 1, 11, 0]
        self.rows["mode"] = [0, 0, 1, 0]
        self.rows["time_signature"] = [4, 4, 3, 4]

    def test_category_mismatches_have_equal_bounded_cost(self):
        coords, _ = knn_coordinates(self.rows.iloc[:3], self.rows.iloc[3:])
        self.assertAlmostEqual(np.sum((coords[0] - coords[1]) ** 2), 1.0)
        self.assertAlmostEqual(np.sum((coords[0] - coords[2]) ** 2), 3.0)
        # Key 11 is no farther from key 0 than key 1 is when other attributes match.
        self.rows.loc[2, ["mode", "time_signature"]] = [0, 4]
        coords, _ = knn_coordinates(self.rows.iloc[:3], self.rows.iloc[3:])
        self.assertAlmostEqual(np.sum((coords[0] - coords[2]) ** 2), 1.0)

    def test_audio_only_matches_original_coordinates(self):
        audio = self.rows[KNN_SPACE].copy()
        audio["duration_ms"] = np.log1p(audio["duration_ms"])
        audio["instrumentalness"] = np.log10(audio["instrumentalness"] + 1e-6)
        expected = ((audio - audio.mean()) / audio.std()).fillna(0).to_numpy()
        tr, te = knn_coordinates(self.rows.iloc[:3], self.rows.iloc[3:], musical=False)
        np.testing.assert_allclose(np.vstack([tr, te]), expected, rtol=0, atol=0)

    def test_missing_and_unseen_test_categories_are_finite(self):
        self.rows.loc[0, "key"] = np.nan
        self.rows.loc[3, "time_signature"] = 7
        tr, te = knn_coordinates(self.rows.iloc[:3], self.rows.iloc[3:])
        self.assertEqual(tr.shape[1], te.shape[1])
        self.assertTrue(np.isfinite(np.vstack([tr, te])).all())

    def test_musical_match_changes_neighbor_but_keeps_date_filter(self):
        self.rows.loc[:, KNN_SPACE] = self.rows.loc[0, KNN_SPACE].to_numpy()
        self.rows.loc[:, "key"] = [0, 1, 1, 1]
        self.rows.loc[:, "mode"] = 0
        self.rows.loc[:, "time_signature"] = 4
        tr, te = knn_coordinates(self.rows.iloc[:3], self.rows.iloc[3:])
        dates = np.array(["2000-01-01", "2000-01-01", "2001-01-01"])
        mean, dist = knn_date(tr, dates, np.array([0., 1., 0.]), te,
                              np.array(["2000-01-01"]), k=1, prior=0.)
        self.assertEqual(mean[0], 1.)
        self.assertEqual(dist[0], 0.)


if __name__ == "__main__":
    unittest.main()
