import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

class ModelTests(unittest.TestCase):
    def test_classification_fixture_and_holdout(self):
        data = pd.read_csv(ROOT / "data/bank_loan_data.csv")
        self.assertFalse(data.isna().any().any())
        self.assertEqual(set(data["Approved"]), {0, 1})
        X = data.drop(columns="Approved")
        train_x, test_x, train_y, test_y = train_test_split(X, data["Approved"], test_size=.25, stratify=data["Approved"], random_state=42)
        model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(train_x, train_y)
        baseline = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
        self.assertGreaterEqual(accuracy_score(test_y, model.predict(test_x)), accuracy_score(test_y, baseline.predict(test_x)))
        self.assertLessEqual(model.get_depth(), 3)

    def test_regression_fixture_and_holdout(self):
        data = pd.read_csv(ROOT / "data/house_prices_data.csv")
        self.assertTrue(np.isfinite(data.to_numpy()).all())
        train_x, test_x, train_y, test_y = train_test_split(data.drop(columns="Price_k$"), data["Price_k$"], test_size=.25, random_state=42)
        model = DecisionTreeRegressor(max_depth=3, random_state=42).fit(train_x, train_y)
        baseline = DummyRegressor().fit(train_x, train_y)
        self.assertLess(mean_squared_error(test_y, model.predict(test_x)), mean_squared_error(test_y, baseline.predict(test_x)))
        self.assertLessEqual(len(np.unique(model.predict(test_x))), 8)
