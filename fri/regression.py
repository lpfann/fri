from fri.base import FRIBase
from fri.l1models import L1EpsilonRegressor
from fri.optproblems import BaseRegressionProblem
from sklearn.utils import check_X_y


class FRIRegression(FRIBase):
    problemType = BaseRegressionProblem

    def __init__(self, C=None, epsilon=None, optimum_deviation=0.001, random_state=None,
                    parallel=False, n_resampling=3, debug=False):
        super().__init__(isRegression=True, C=C, random_state=random_state,
                         parallel=parallel,
                         n_resampling=n_resampling,
                         debug=debug, optimum_deviation=optimum_deviation)
        self.epsilon = epsilon
        self.initModel = L1EpsilonRegressor


    def fit(self, X, y):
        """ Fit model to data and provide feature relevance intervals
        Parameters
        ----------
        X : array_like
            standardized data matrix
        y : array_like
            response vector
        """
        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}
        # Only use parameter grid when no parameter is given
        if self.C is None:
            self.tuned_parameters["C"] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        else:
            self.tuned_parameters["C"] = [self.C]

        if self.epsilon is None:
            self.tuned_parameters["epsilon"] = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        else:
            self.tuned_parameters["epsilon"] = [self.epsilon]

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        super().fit(X, y)