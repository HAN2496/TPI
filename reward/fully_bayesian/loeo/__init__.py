"""Leave-one-evaluator-out (LOEO) experiment package for the T-IV draft.

Modules
-------
bank        candidate feature bank per channel and correlation pruning
data        real (loader) and synthetic data sources with a common interface
metrics     proper scores, calibration, uncertainty-task metrics, prequential score
protocol    folds, budgets, offsets and per-evaluator evaluation of the proposed model
baselines   pooled / independent logistic, empirical-Bayes MAP, gradient boosting
selection   projection-predictive path evaluation, one-SE rule, sensor subsets, stability
report      aggregation over folds into tables and figures
"""
