# Global and instrumental-head neural networks

Recommendation: retain the existing tree ensemble. The instrumental head improves its subgroup on average, but worsens overall weighted AUC in all five seeds. Neither fixed 10% NN blend improves the tree baseline. This is evidence against adopting this configuration, not against every possible neural architecture.

Both networks use a 64/32 ReLU backbone, dropout 0.1 and AdamW (learning rate 0.001, weight decay 0.0001). The specialist adds a 32/8/1 residual branch with a zero-initialized last layer. Instrumentalness >= 0.5 activates it; missing values use the general head.

The general BCE loss uses all rows. The specialist adds 0.25 times instrumental routed BCE plus 0.01 times mean squared instrumental residual logits. Each minibatch averages its own eligible rows. No class weighting or oversampling. Settings were fixed before examining results.

Five seeds, five URI-grouped outer folds per seed, with a separate grouped inner split selecting epochs (maximum 150, patience 15, minimum logloss improvement 0.0001). Each model is then freshly trained on all outer-training rows for its selected epoch count: 50 selection fits and 50 final fits. Outer labels never select epochs. Identical conflicting-label tracks remain grouped and retain their labels.

All imputation, scaling and category learning use permitted training rows only. Date target encodings and audio-only same-date KNN label features use five-fold URI-grouped cross-fitting for training rows. Validation/test features use the corresponding training references only. Musical attributes are one-hot NN inputs. File-level track/date counts are omitted. These preprocessing differences mean comparisons with older trees are not architecture-only comparisons.

## Mean per-seed OOF results

```
                                    auc  weighted_auc  logloss    brier  high_miss_rate
model             group                                                                
global            instrumental 0.730136      0.721294 0.198296 0.049492        0.931915
                  other        0.770071      0.764629 0.567291 0.192037        0.396799
                  overall      0.822037      0.819381 0.488200 0.161484        0.415305
instrumental_head instrumental 0.739606      0.731074 0.197199 0.049367        0.927660
                  other        0.767391      0.761588 0.569653 0.193023        0.399390
                  overall      0.820499      0.817496 0.489820 0.162232        0.417660
```

Overall head-minus-global weighted AUC: -0.001885; wins in 0/5 seeds. These overlapping repeated splits are not five independent experiments. Results are exploratory because the outer splits have been reused. High miss rate uses a fixed 0.5 threshold and is secondary to ranking quality.

## Training gaps

```
                   train_auc  validation_auc    epochs  auc_gap
model                                                          
global              0.867649        0.823225 18.680000 0.044424
instrumental_head   0.863867        0.821552 19.680000 0.042315
```

Training AUC uses the cross-fitted training features seen during fitting; validation uses full training references. These gaps are descriptive, not a formal bias–variance decomposition.

## Fixed ensemble comparison

```
                       model      auc  weighted_auc  logloss    brier  weighted_auc_delta
                       trees 0.863475      0.847765 0.500715 0.167564            0.000000
           trees_plus_global 0.863208      0.847645 0.501272 0.167819           -0.000119
trees_plus_instrumental_head 0.863081      0.847473 0.501380 0.167879           -0.000292
```

The existing four tree models retain equal rank weights. Average probabilities over seeds first, rank each model, then blend 90% existing tree ranks and 10% NN ranks. No weight search. This aggregation differs from mean per-seed AUC above; rank blends are ranking scores, not calibrated probabilities. Existing default models and submissions are unchanged.

Artifacts: predictions.npz contains per-seed OOF/test arrays, labels, weights, source rows and shared-head components; CSV files contain row-aligned predictions, split memberships, all epoch losses, paired deltas and four positive/negative subgroup AUC contributions. Submission IDs match the original test IDs. Model weights are not persisted; rerun the script to refit.

Reproduce from repository root with `.venv/Scripts/python.exe final_pipeline/nn_instrumental.py --out <fresh-directory>`. Dependencies are pinned in `final_pipeline/nn-requirements-lock.txt`; install CPU torch from https://download.pytorch.org/whl/cpu. The report script targets the dated experiment directory. Four focused unit tests and post-run checks cover group separation, query-label/statistic isolation, prediction ordering, zero-initialized residuals, masked routing, gradients, finite outputs and pairwise AUC reconciliation.
