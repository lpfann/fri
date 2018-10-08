import numpy as np
from sklearn.utils import check_X_y

from fri.base import FRIBase
from fri.l1models import L1OrdinalRegressor
from fri.optproblems import BaseOrdinalRegressionProblem


class FRIOrdinalRegression(FRIBase):

    """
        Class for ordinal regression data
    """

    problemType = BaseOrdinalRegressionProblem

    def __init__(self, C=None, random_state=None, optimum_deviation=0.001,
                 parallel=False, debug=False, **kwargs):
        super().__init__(isRegression=False, C=C, random_state=random_state,
                         parallel=parallel,
                         debug=debug,
                         optimum_deviation=optimum_deviation,
                         **kwargs)

        self.initModel = L1OrdinalRegressor

        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}

        # Only use parameter grid when no parameter is given
        # TODO: Set appropraite values for C
        if self.C is None:
            self.tuned_parameters["C"] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        else:
            self.tuned_parameters["C"] = [self.C]


    def fit(self, X, y):

        """
        Fit model to data and provide feature relevance intervals
        Parameters
        ----------
        X : array_like
            standardized data matrix
        y : array_like
            response vector
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Get ordinal classes
        self.classes_ = np.unique(y)

        super().fit(X, y)