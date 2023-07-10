"""
A singleton for holding the model. It loads the model.
The predict function predicts based on the input and model.
"""
# pylint: disable=E0401
import os

import joblib
from explore_ai_demo.preprocess import main, woevars

PREFIX = "/opt/ml"
model_path = os.path.join(PREFIX, "model")


def extract_relevant_columns(data, woevars):
    return data[woevars]


class P2PModel:
    """
    Propensity to Pay model.
    """

    model = None

    def load(self):
        """
        Load model.
        :return:
        """
        if self.model is None:
            self.model = joblib.load(os.path.join(model_path, "model.joblib"))
        return self.model

    @staticmethod
    def preprocess(data_frame):
        """
        Preprocess features
        :return: dataframe
        """
        return main(data_frame)

    def predict(self, data):
        """
        Predict.
        """

        all_data = self.preprocess(data)

        data = extract_relevant_columns(all_data, woevars)

        return {
            "probability": self.model.predict_proba(data)[:, 1],
            "class": self.model.predict(data),
        }
