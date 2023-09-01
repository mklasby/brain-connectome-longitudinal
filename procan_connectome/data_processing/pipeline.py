from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import RFECV, SelectFromModel  # noqa
from sklearn.base import clone

from ingeniator.feature_selection.sklearn_transformer_wrapper import (
    SklearnTransformerWrapper,
)
from ingeniator.feature_selection.feature_selection_transformer import (
    FeatureSelectionTransformer,
)


def get_pipeline(cfg, estimator):
    steps = []
    if cfg.pipeline.standard_scale:
        steps.append(
            SklearnTransformerWrapper(
                transformer=StandardScaler(),
                ignore_features=cfg.pipeline.ignore_features,
            )
        )
    if cfg.pipeline.power_transform:
        steps.append(
            SklearnTransformerWrapper(
                transformer=PowerTransformer(standardize=False),
                ignore_features=cfg.pipeline.ignore_features,
            )
        )
    if cfg.pipeline.rfecv:
        steps.append(
            FeatureSelectionTransformer(
                transformer=RFECV(
                    estimator=clone(estimator),
                    n_jobs=cfg.n_jobs,
                    step=0.1,
                    verbose=10,
                    min_features_to_select=10,
                ),
                ignore_features=cfg.pipeline.ignore_features,
            )
        )
    pipe = make_pipeline(*steps)
    if len(pipe.steps) < 1:
        pipe = None
    return pipe
