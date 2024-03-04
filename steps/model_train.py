import logging
import pandas as pd
from zenml import step

import mlflow
from source.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig

from .config  import ModelNameConfig
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
    ) -> RegressorMixin:
    """Trains the model on the ingested data

    Args:
        df (pd.DataFrame): _description_
    """
    try:
        model = None
        if config.model_name == "Linear Regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
