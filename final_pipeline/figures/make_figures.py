"""Report figures (static PNG for LaTeX).

Run from anywhere:  python final_pipeline/figures/make_figures.py
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train import TARGET, date_features

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
OUT = HERE
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
tr = pd.read_csv(os.path.join(DATA, "train.csv")).dropna(subset=["uri"]).reset_index(drop=True)
tr["y"] = (tr[TARGET] == "High").astype(int)
y = tr["y"].values
d = date_features(tr)

# ---- Figure 1: High rate by release era
bins = [1900, 1980, 1990, 2000, 2010, 2015, 2018, 2020, 2022, 2023, 2024, 2025]
labels = ["<1980", "80s", "90s", "00s", "10-14", "15-17", "18-19", "20-21", "2022", "2023", "2024"]
g = pd.DataFrame({"y": y, "b": pd.cut(d["rel_year"], bins, labels=labels, right=False)}).groupby("b", observed=True)["y"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(g.index.astype(str), g["mean"], color=BLUE, width=0.62)
for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
    ax.text(i, m + 0.02, f"n={n}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
ax.axhline(y.mean(), color=MUTED, lw=1, ls="--"); ax.text(10.4, y.mean() + 0.015, f"overall {y.mean():.2f}", ha="right", fontsize=8, color=MUTED)
ax.set(ylabel="share of High-popularity tracks", xlabel="album release year", ylim=(0, 1.08), title="Older tracks that survive into 2024 playlists are almost all 'High'")
ax.grid(axis="x", visible=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig_year_rate.png", dpi=200); plt.close(fig)

# ---- Figure 2: track multiplicity and album effect
cnt = tr.groupby("uri")["uri"].transform("count")
g2 = pd.DataFrame({"y": y, "c": cnt}).groupby("c")["y"].agg(["mean", "count"])
u = tr.drop_duplicates("uri").reset_index(drop=True); yu = u["y"].values
AUD = ["speechiness", "danceability", "energy", "instrumentalness", "valence", "loudness", "acousticness", "liveness", "tempo", "duration_ms"]
Xa = u[AUD].copy(); Xa["duration_ms"] = np.log1p(Xa["duration_ms"]); Xa["instrumentalness"] = np.log10(Xa["instrumentalness"] + 1e-6)
Xa = ((Xa - Xa.mean()) / Xa.std()).values
dates = u["track_album_release_date"].astype(str).values; year = pd.to_numeric(pd.Series(dates).str[:4]).values
groups = {}
for i, dd in enumerate(dates):
    groups.setdefault(dd, []).append(i)
rng = np.random.default_rng(0); agree_nn, agree_rand, agree_any = [], [], []
for i, dd in enumerate(dates):
    idx = np.array([j for j in groups[dd] if j != i])
    if len(idx) == 0:
        continue
    dist = np.sqrt(((Xa[idx] - Xa[i]) ** 2).sum(1)); j = idx[np.argmin(dist)]
    agree_nn.append(yu[j] == yu[i])
    sy = np.where(year == year[i])[0]; sy = sy[sy != i]; agree_rand.append(yu[rng.choice(sy)] == yu[i])
    agree_any.append(yu[rng.choice(np.arange(len(yu)))] == yu[i])
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax = axes[0]
ax.bar([str(c) for c in g2.index], g2["mean"], color=BLUE, width=0.55)
for i, (m, n) in enumerate(zip(g2["mean"], g2["count"])):
    ax.text(i, m + 0.02, f"{m:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8, color=MUTED)
ax.set(xlabel="times the track appears in train.csv (uri_count)", ylabel="share High", ylim=(0, 1.2), title="Track multiplicity")
ax.grid(axis="x", visible=False)
ax = axes[1]
vals = [np.mean(agree_any), np.mean(agree_rand), np.mean(agree_nn)]
ax.barh(["random track", "random same-year track", "nearest same-date track\n(album-mate proxy)"], vals, color=[MUTED, MUTED, ORANGE], height=0.5)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
ax.set(xlim=(0.5, 0.8), xlabel="label agreement with the comparison track", title=f"Album effect (n={len(agree_nn)} tracks)")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_multiplicity_album.png", dpi=200); plt.close(fig)

# ---- Figure 3: ablation / progression (grouped 5-fold CV, mean of 3 seeds)
rows = [("audio features only (LightGBM)", 0.7517, None), ("+ release-date features", 0.8325, 0.8292),
        ("+ uri_count", 0.8497, 0.8353), ("+ album target-enc + same-date kNN", 0.8536, 0.8375),
        ("+ Optuna tuning (LightGBM)", 0.8571, 0.8410), ("rank-average blend (LGB+XGB+Cat+ET)", 0.8595, 0.8433)]
fig, ax = plt.subplots(figsize=(7.4, 3.3))
names = [r[0] for r in rows][::-1]; plain = [r[1] for r in rows][::-1]; tl = [r[2] if r[2] else np.nan for r in rows][::-1]
yy = np.arange(len(rows))
ax.barh(yy + 0.18, plain, height=0.34, color=BLUE, label="OOF AUC")
ax.barh(yy - 0.18, tl, height=0.34, color=ORANGE, label="test-like OOF AUC (duplicates down-weighted)")
for i, (p, t) in enumerate(zip(plain, tl)):
    ax.text(p + 0.002, i + 0.18, f"{p:.3f}", va="center", fontsize=8, color=INK)
    if not np.isnan(t):
        ax.text(t + 0.002, i - 0.18, f"{t:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9); ax.set(xlim=(0.70, 0.90), xlabel="grouped 5-fold CV AUC (mean of seeds)", title="What moved the score")
ax.axvline(0.837, color=MUTED, lw=1, ls="--"); ax.text(0.8385, len(rows) - 0.95, "previous best\nensemble (0.837)", fontsize=7.5, color=MUTED, va="top")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, fontsize=8, frameon=False); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.png", dpi=200); plt.close(fig)
print("saved", os.listdir(OUT))
