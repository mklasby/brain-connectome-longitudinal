import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.ensemble import BalancedRandomForestClassifier


def get_estimator_and_grid(cfg):
    model_name = cfg.model.lower()
    models = {"svc": _svc, "rf": _rf, "xgb": _xgb, "brfc": _brfc}
    return models[model_name](cfg)


def _svc(cfg):
    estimator = LinearSVC(
        C=0.1,
        fit_intercept=True,
        max_iter=int(1e9),
        tol=1e-10,
        random_state=cfg.random_state,
        class_weight="balanced",
    )
    grid = {
        "tol": [1e-6],
        "C": [0.0001, 0.005, 0.1, 0.5],
        "fit_intercept": [True, False],
        "max_iter": [10000],
        "class_weight": ["balanced", None],
    }

    return estimator, grid


def _rf(cfg):
    estimator = RandomForestClassifier(random_state=cfg.random_state)
    grid = {
        "n_estimators": [50, 100, 200, 400, 600],
        "criterion": ["entropy", "gini"],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1],
        "class_weight": ["balanced", None],
    }
    return estimator, grid


def _xgb(cfg):
    estimator = xgb.XGBClassifier(
        objective="multi:softmax",
        random_state=cfg.random_state,
        eval_metric="auc",
        num_class=5,
        n_estimators=200,
        max_depth=7,
        learning_rate=0.2,
    )
    grid = {
        "max_depth": [None, 3, 5],
        "n_estimators": [50, 100, 200, 400],
        "gamma": [None, 0.1],
        "reg_lambda": [None, 0.1, 1.0],
        "alpha": [None, 0.1],
        "learning_rate": [None, 0.1, 0.2],
    }
    return estimator, grid


def _brfc(cfg):
    estimator = BalancedRandomForestClassifier(random_state=cfg.random_state)
    grid = {
        "n_estimators": [100, 200, 400, 600],
        "criterion": ["entropy", "gini"],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1],
        "class_weight": ["balanced", None],
    }
    return estimator, grid
