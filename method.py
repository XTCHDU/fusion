from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import tensorflow as tf


class ML:
    def __init__(self):
        self.model = GradientBoostingRegressor()
        self.model = MultiOutputRegressor(self.model)

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def mse(x, y):
        return mean_squared_error(x, y)

    def save(self, model_file):
        joblib.dump(self.model, model_file)

    def load(self, model_file):
        self.model = joblib.load(model_file)

