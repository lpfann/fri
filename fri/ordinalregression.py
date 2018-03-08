from sklearn.utils import check_X_y


from fri.base import FRIBase
from fri.l1models import L1OrdinalRegressor
from fri.optproblems import MinProblemOrdinalRegression, MaxProblem1OrdinalRegression, MaxProblem2OrdinalRegression


class FRIOrdinalRegression(FRIBase):

    """
        Class for ordinal regression data
    """

    minProblem = MinProblemOrdinalRegression
    maxProblem1 = MaxProblem1OrdinalRegression
    maxProblem2 = MaxProblem2OrdinalRegression

    def __init__(self, epsilon=None, C=None, random_state=None,
                 shadow_features=False, parallel=False, feat_elim=False, **kwargs):
        super().__init__(isRegression=False, C=C, random_state=random_state,
                         shadow_features=shadow_features, parallel=parallel, feat_elim=False, **kwargs)

        self.initModel = L1OrdinalRegressor

        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}

        # Only use parameter grid when no parameter is given
        # TODO: Set appropraite values for C and delta
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

        super().fit(X, y)
