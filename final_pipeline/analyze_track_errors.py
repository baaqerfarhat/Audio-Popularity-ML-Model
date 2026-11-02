"""Track-level consensus errors from saved, group-held-out predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def main():
    here = Path(__file__).resolve().parent
    out = here / "experiments/track_errors_2026_11_01"
    out.mkdir(parents=True, exist_ok=True)
    source = here / "experiments/bias_variance_2026_11_01"
    raw = pd.read_csv(here.parent / "data/train.csv")
    raw["csv_line"] = raw.index + 2
    tr = raw.dropna(subset=["uri"]).reset_index(drop=True)
    tr["source_row"] = tr.index
    tr["y"] = tr.Popularity_Type.eq("High").astype(int)
    models = ["lgb_tuned", "lgb_deep", "xgb_tuned", "cat_tuned", "cat_base", "extra_trees"]
    preds = pd.read_csv(source / "oof_predictions.csv", float_precision="round_trip")
    preds = preds[preds.model.isin(models)].copy()
    assert len(preds) == len(tr)*18 and not preds.duplicated(["source_row", "model", "seed"]).any()
    merged = preds.merge(tr[["source_row", "uri", "y"]], on="source_row", suffixes=("", "_source"), validate="many_to_one")
    assert merged.y.eq(merged.y_source).all()
    conflicts = tr.groupby("uri").y.nunique()
    conflicting = conflicts[conflicts > 1].index
    tr[tr.uri.isin(conflicting)].to_csv(out / "conflicting_labels.csv", index=False)
    conflict_rows = tr[tr.uri.isin(conflicting)]
    fields = [c for c in raw.columns if c not in ["Popularity_Type", "csv_line"]]
    same_fields = conflict_rows.groupby("uri")[fields].nunique(dropna=False).le(1).all(axis=1)
    conflict_summary = conflict_rows.groupby(["uri", "Popularity_Type"]).size().unstack(fill_value=0)
    conflict_summary["identical_all_input_fields"] = same_fields
    conflict_summary.reset_index().to_csv(out / "conflict_summary.csv", index=False)
    good = tr[~tr.uri.isin(conflicting)].copy()
    # Each unique URI receives one prediction per model/seed, irrespective of row multiplicity.
    wide = merged[~merged.uri.isin(conflicting)].groupby(["uri", "model", "seed"]).probability.mean().unstack(["model", "seed"])
    assert wide.shape[1] == 18 and np.isfinite(wide.to_numpy()).all()
    tracks = good.groupby("uri", sort=True).first().reindex(wide.index)
    tracks["row_count"] = good.groupby("uri").size()
    tracks["csv_lines"] = good.groupby("uri").csv_line.apply(lambda s: ",".join(s.astype(str)))
    p = wide.to_numpy(); y = tracks.y.to_numpy()
    wrong = (p >= .5) != y[:, None]
    tracks["wrong_votes"] = wrong.sum(axis=1)
    tracks["persistent_wrong"] = wrong.all(axis=1)
    tracks["mean_p_high"] = p.mean(axis=1)
    tracks["min_p_high"] = p.min(axis=1)
    tracks["max_p_high"] = p.max(axis=1)
    tracks["prediction_sd"] = p.std(axis=1, ddof=1)
    correct_p = np.where(y[:, None] == 1, p, 1-p)
    tracks["mean_logloss"] = -np.log(np.clip(correct_p, np.finfo(float).eps, 1)).mean(axis=1)
    tracks["all_confident_wrong"] = (correct_p <= .2).all(axis=1)
    # Rank score and per-track pairwise errors target the AUC objective without a 0.5 cutoff.
    scores = np.apply_along_axis(rankdata, 0, p).mean(axis=1)/len(tracks)
    tracks["consensus_rank_score"] = scores
    burden = np.zeros(len(tracks))
    for label in [0, 1]:
        opposite = np.sort(scores[y != label]); current = scores[y == label]
        lo = np.searchsorted(opposite, current, side="left")
        hi = np.searchsorted(opposite, current, side="right")
        burden[y == label] = ((len(opposite)-hi if label else lo) + .5*(hi-lo))/len(opposite)
    tracks["opposite_class_misranking_fraction"] = burden
    tracks["year"] = pd.to_numeric(tracks.track_album_release_date.astype(str).str[:4], errors="coerce")
    tracks["release_period"] = pd.cut(tracks.year, [-np.inf, 1999, 2009, 2019, np.inf],
                                      labels=["Before 2000", "2000–2009", "2010–2019", "2020 onward"]).astype(str)
    tracks["multiplicity"] = np.where(tracks.row_count > 1, "Repeated URI", "Single row")
    tracks["instrumental_group"] = np.where(tracks.instrumentalness >= .5, "Instrumentalness >= 0.5", "Instrumentalness < 0.5")
    membership = pd.read_csv(source / "fold_membership.csv")
    mem = membership.merge(tr[["source_row", "uri", "track_album_release_date"]], on="source_row", validate="many_to_one")
    assert mem.groupby(["seed", "uri"]).fold.nunique().eq(1).all()
    coverage = []
    for seed, frame in mem.groupby("seed"):
        frame = frame.drop_duplicates("uri")
        for fold in range(5):
            training = frame[frame.fold != fold]
            validation = frame[frame.fold == fold].copy()
            counts = training.groupby("track_album_release_date").uri.nunique()
            validation["same_date_train_tracks"] = validation.track_album_release_date.map(counts).fillna(0)
            coverage.append(validation[["uri", "seed", "same_date_train_tracks"]])
    coverage = pd.concat(coverage)
    tracks["mean_same_date_train_tracks"] = coverage.groupby("uri").same_date_train_tracks.mean()
    tracks["date_neighbor_coverage"] = np.where(coverage.groupby("uri").same_date_train_tracks.max().reindex(tracks.index).eq(0),
                                                "Never has date neighbors", "Has date neighbors in some/all splits")
    tracks.reset_index().to_csv(out / "all_tracks.csv", index=False)
    hard = tracks[tracks.persistent_wrong].sort_values("mean_logloss", ascending=False)
    hard.reset_index().to_csv(out / "persistent_errors.csv", index=False)
    top = pd.concat([hard[hard.y.eq(label)].head(10) for label in [1, 0]])
    top.reset_index().to_csv(out / "top_examples.csv", index=False)
    metadata_path = out / "spotify_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lookup = pd.DataFrame([dict(uri=e["uri"], track_name=e["name"],
                                    artists=", ".join(a["name"] for a in e["creators"]), spotify_url=e["url"])
                               for e in metadata["entities"]])
        top.reset_index().merge(lookup, on="uri", how="left", validate="one_to_one").to_csv(out / "top_examples.csv", index=False)
    tracks.groupby(["Popularity_Type", "persistent_wrong"])[
        ["instrumentalness", "acousticness", "speechiness", "energy", "danceability", "loudness"]
    ].median().to_csv(out / "feature_medians.csv")
    (out / "lookup_uris.json").write_text(json.dumps(top.index.tolist(), indent=2))
    segments = []
    for feature in ["release_period", "multiplicity", "date_neighbor_coverage", "instrumental_group"]:
        for (value, label), group in tracks.groupby([feature, "Popularity_Type"]):
            segments.append(dict(feature=feature, segment=value, label=label, tracks=len(group),
                                 persistent_errors=int(group.persistent_wrong.sum()),
                                 persistent_error_rate=group.persistent_wrong.mean(),
                                 mean_misranking_fraction=group.opposite_class_misranking_fraction.mean()))
    segments = pd.DataFrame(segments)
    segments.to_csv(out / "segments.csv", index=False)
    counts = dict(training_rows=len(tr), unique_tracks=int(tr.uri.nunique()), excluded_conflicting_uris=len(conflicting),
                  conflicting_rows=len(conflict_rows), identical_input_conflicting_uris=int(same_fields.sum()),
                  analyzed_tracks=len(tracks), predictions_per_track=18, persistent_errors=len(hard),
                  high_tracks=int(tracks.y.sum()), low_tracks=int((tracks.y==0).sum()),
                  missed_high=int(hard.y.sum()), false_high=int((hard.y==0).sum()),
                  all_confident_wrong=int(tracks.all_confident_wrong.sum()),
                  consensus_unique_track_auc=float(1-burden[y==1].mean()),
                  source_hashes={str(path.relative_to(here.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [here.parent / "data/train.csv", source / "oof_predictions.csv", source / "fold_membership.csv"]})
    (out / "summary.json").write_text(json.dumps(counts, indent=2)+"\n")
    print(json.dumps(counts, indent=2))
    print(top[["Popularity_Type", "mean_p_high", "min_p_high", "max_p_high", "year", "row_count", "opposite_class_misranking_fraction"]].to_string())
    print(segments.to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    periods = ["Before 2000", "2000–2009", "2010–2019", "2020 onward"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for label, offset, color in [("High", -.18, "#28649B"), ("Low", .18, "#B07113")]:
        d = segments[(segments.feature=="release_period") & (segments.label==label)].set_index("segment").reindex(periods)
        bars=ax.bar(np.arange(4)+offset, 100*d.persistent_error_rate, .34, color=color,
                    label="High labeled, always predicted Low" if label=="High" else "Low labeled, always predicted High")
        for bar, (_, row) in zip(bars, d.iterrows()):
            ax.annotate(f"{int(row.persistent_errors)}/{int(row.tracks)}", (bar.get_x()+bar.get_width()/2,bar.get_height()),
                        xytext=(0,4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(np.arange(4), periods)
    ax.set(ylabel="Persistent errors within each label group (%)", title="Tracks misclassified in all 18 held-out predictions")
    ax.set_ylim(0, max(segments[segments.feature.eq('release_period')].persistent_error_rate)*100+10)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(.5,.01,"Unique tracks; six model variants × three seeds; diagnostic cutoff 0.5. Labels show errors / tracks.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(out / "error_patterns.png",dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
