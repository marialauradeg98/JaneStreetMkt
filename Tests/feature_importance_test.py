import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from JaneStreetMkt.f_feature_importance import compute_feat_imp
from sklearn.ensemble import ExtraTreesClassifier


class TestImport(unittest.TestCase):

    def test_compute_importance(self):

        # load load breast_cancer dataset
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # create and fit model
        forest = ExtraTreesClassifier()
        forest.fit(X, y)

        # create array of columns names
        columns_data = np.zeros(X.shape[1])
        columns_data = X.columns

        # compute importance
        red = redundant_feat = compute_feat_imp(forest, columns_data)


if __name__ == "__main__":
    unittest.main()
