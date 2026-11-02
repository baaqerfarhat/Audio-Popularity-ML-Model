# Release-period specialists on the 10-fold pipeline (period_specialists_10fold_2026_11_02)

Specialised component: `lgb_tuned`; periods: release year < 2020 / >= 2020; unknown year -> global. 5 seeds x 10 grouped folds; feature set `v2`, ExtraTrees `base`. Primary metric: mean per-seed weighted (test-like) OOF AUC. Exploratory: the validation splits were reused.

## Single component (mean of per-seed pooled OOF scores)

| variant | cohort | AUC | weighted AUC | weighted delta vs global | seed wins |
|---|---|---:|---:|---:|---:|
| global | overall | 0.859779 | 0.843541 | +0.000000 | 0/5 |
| global | before_2020 | 0.842865 | 0.824670 | +0.000000 | 0/5 |
| global | 2020_onward | 0.844789 | 0.820935 | +0.000000 | 0/5 |
| specialist | overall | 0.857037 | 0.840798 | -0.002743 | 0/5 |
| specialist | before_2020 | 0.835252 | 0.817035 | -0.007635 | 0/5 |
| specialist | 2020_onward | 0.844398 | 0.820377 | -0.000558 | 1/5 |
| mix | overall | 0.860394 | 0.844397 | +0.000856 | 4/5 |
| mix | before_2020 | 0.841431 | 0.823467 | -0.001203 | 1/5 |
| mix | 2020_onward | 0.846539 | 0.822952 | +0.002018 | 5/5 |

## Cross-period score compatibility (weighted pair AUC, mean over seeds)

| High period / Low period | global | specialist | mix |
|---|---:|---:|---:|
| before_2020 / before_2020 | 0.824670 | 0.817035 | 0.823467 |
| before_2020 / 2020_onward | 0.935018 | 0.935420 | 0.937571 |
| 2020_onward / before_2020 | 0.624872 | 0.613194 | 0.620163 |
| 2020_onward / 2020_onward | 0.820935 | 0.820377 | 0.822952 |

Mean predicted probability by period (score-scale drift check):

| variant    |   2020_onward |   before_2020 |
|:-----------|--------------:|--------------:|
| global     |        0.2527 |        0.4992 |
| mix        |        0.2479 |        0.5021 |
| specialist |        0.2431 |        0.505  |

## Fixed four-member ensemble (equal rank average; only the specialised component changes)

Seed-averaged probabilities -> rank -> average (submission aggregation):

| component variant | AUC | weighted AUC | weighted delta |
|---|---:|---:|---:|
| global | 0.864634 | 0.848671 | +0.000000 |
| specialist | 0.864681 | 0.848777 | +0.000106 |
| mix | 0.864750 | 0.848843 | +0.000173 |

Mean per-seed AUC of per-seed rank blends (paired):

| component variant | mean AUC | mean weighted AUC | weighted delta | seed wins |
|---|---:|---:|---:|---:|
| global | 0.862315 | 0.846109 | +0.000000 | 0/5 |
| specialist | 0.862430 | 0.846293 | +0.000184 | 3/5 |
| mix | 0.862514 | 0.846357 | +0.000249 | 5/5 |

Candidate submissions: `submission_fixed_blend_mix.csv`, `submission_fixed_blend_specialist.csv`, `submission_lgb_tuned_mix.csv` (baseline files: `submission_fixed_blend_global.csv`, `submission_lgb_tuned_global.csv`).
