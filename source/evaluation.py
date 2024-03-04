import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract class defining strategies for evluationg our models
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the score of the model

        Args:
            y_true (np.ndarray): True Labels
            y_pred (np.ndarray): Predicted Labels
        """
        None

class MSE(Evaluation):
    """Evaluation strategy that uses Mean squared error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in Calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """Evaluation strategy that uses R2 score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true,y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in Calculating R2 score: {}".format(e))
            raise e

class RMSE(Evaluation):
    """Evaluation strategy that uses RMSE score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try: 
            logging.info("Calculating RMSE score")
            rmse = mean_squared_error(y_true,y_pred,squared=False)
            logging.info("RMSE Score: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in Calculating RMSE score: {}".format(e))
            raise e