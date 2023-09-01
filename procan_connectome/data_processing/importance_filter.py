# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: 'Python 3.7.10 64-bit (''ML'': conda)'
#     name: python3
# ---


import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from procan_connectome.config import RANDOM_STATE
from procan_connectome.data_processing.linear_svc_importance_filter import (
    LinearSVCImportanceFilter,
)
import logging


# +
def svc_loovc(
    X: pd.DataFrame,
    y: pd.Series,
    standard_scale: bool = False,
    power_transform: bool = False,
    threshold: float = 0.001,
):
    loo = LeaveOneOut()
    importances = []
    y_true = []
    y_predict = []
    counter = 0

    for train_idx, test_idx in loo.split(X):
        counter += 1
        if counter % 10 == 0:
            print(f"Iteration {counter} of {len(X)}")
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        if standard_scale:
            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)
        if power_transform:
            pt = PowerTransformer()
            X_train = pt.fit_transform(X_train)
            X_test = pt.transform(X_test)
        svc = LinearSVCImportanceFilter(
            random_state=RANDOM_STATE, sort=False, threshold=threshold
        )
        svc.fit(X_train, y_train)
        importances.append(svc.feature_importances_df_["Importance"].values)
        y_predict.append(svc.estimator_.predict(X_test)[0])
        y_true.append(y_test[0])

    acc = accuracy_score(y_true, y_predict)
    logging.debug(f"Feature Selection LOOCV Accuracy: {acc}")
    labels = X.columns
    feature_importances = np.array(importances).mean(axis=0)
    feature_importances_df = pd.DataFrame(
        list(zip(map(lambda x: round(x, 6), feature_importances), labels)),
        columns=["Importance", "Feature"],
    )
    logging.debug(f"First 10 Features: {feature_importances_df.iloc[:10]}")
    results_df = pd.DataFrame({"y_true": y_true, "y_pred": y_predict}).set_index(
        X.index
    )
    return results_df, feature_importances_df
