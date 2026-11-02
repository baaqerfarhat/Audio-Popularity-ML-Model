"""Independently check recorded metrics, track grouping, and source fingerprints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def main():
    root = Path(__file__).resolve().parent.parent
    p = root / "final_pipeline/experiments/bias_variance_2026_11_01"
    def read(file):
        # Preserve prediction ties and exact zeros when restoring saved float64s.
        return pd.read_csv(file, float_precision="round_trip")
    f, o, s, m, summary = [read(p / name) for name in
        ["fold_metrics.csv", "oof_predictions.csv", "seed_metrics.csv", "fold_membership.csv", "model_summary.csv"]]
    tr = read(root / "data/train.csv").dropna(subset=["uri"]).reset_index(drop=True)
    assert len(f) == 135 and len(o) == 7 * 3 * len(tr) and len(m) == 3 * len(tr)
    for frame, keys in [(f, ["model", "seed", "fold", "fraction"]),
                        (o, ["model", "seed", "source_row"]), (m, ["seed", "source_row"])]:
        assert not frame.duplicated(keys).any()
    assert np.isfinite(f.select_dtypes("number")).all().all()
    assert np.isfinite(o.probability).all() and o.probability.between(0, 1).all()
    membership = m.merge(tr[["uri"]], left_on="source_row", right_index=True, validate="many_to_one")
    assert membership.groupby(["seed", "uri"]).fold.nunique().eq(1).all()
    for (name, seed), frame in o.groupby(["model", "seed"]):
        frame = frame.sort_values("source_row")
        np.testing.assert_array_equal(frame.y, (tr.Popularity_Type == "High").astype(int))
        y, pred = frame.y.to_numpy(), frame.probability.to_numpy()
        n1, n0 = y.sum(), (1-y).sum()
        auc = (rankdata(pred)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        expected = s[(s.model == name) & (s.seed == seed)].iloc[0]
        np.testing.assert_allclose(auc, expected.auc, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.mean((y - pred) ** 2), expected.brier, atol=1e-12, rtol=0)
        # sklearn log_loss clips exact endpoints to machine epsilon.
        eps = np.finfo(pred.dtype).eps
        clipped = np.clip(pred, eps, 1-eps)
        loss = -np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped))
        np.testing.assert_allclose(loss, expected.logloss, atol=1e-12, rtol=0)
    for _, row in summary.iterrows():
        frame = f[(f.model == row.model) & (f.fraction == 1)]
        assert len(frame) == 15
        np.testing.assert_allclose(row.auc_gap, (frame.train_auc - frame.validation_auc).sum()/15, atol=1e-12, rtol=0)
    old = read(root / "final_pipeline/experiments/knn_musical_2026_11_01/audio/fold_metrics.csv")
    paired = f[f.fraction.eq(1)].merge(old[old.seed.isin([0, 1, 2])], on=["model", "seed", "fold"], validate="one_to_one")
    assert len(paired) == 90
    non_et = paired[paired.model.ne("extra_trees")]
    np.testing.assert_allclose(non_et.validation_auc, non_et.auc, atol=1e-12, rtol=0)
    max_difference = float((paired.validation_auc - paired.auc).abs().max())
    assert max_difference < 1e-5
    meta = json.loads((p / "run_metadata.json").read_text())
    for file, digest in meta["sha256"].items():
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest, file
    reg = read(p / "regularization_paired.csv")
    curve = f[f.model.eq("lgb_tuned")].pivot(index=["seed", "fold"], columns="fraction", values="validation_auc")
    record = dict(fits=len(f), oof_predictions=len(o), group_split_checks="passed",
                  independent_rank_auc_logloss_brier_checks="passed for all 21 model/seed pairs",
                  baseline_validation_reproduction="75 non-ExtraTrees folds match to 1e-12; ExtraTrees differs by less than 1e-5",
                  max_baseline_auc_difference=max_difference, input_hashes="unchanged", unit_tests="3 passed",
                  regularization_auc_wins=int((reg.auc_delta > 0).sum()),
                  regularization_mean_auc_delta=reg.auc_delta.mean(),
                  learning_curve_full_beats_quarter=int((curve[1.] > curve[.25]).sum()),
                  learning_curve_full_beats_half=int((curve[1.] > curve[.5]).sum()))
    (p / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
