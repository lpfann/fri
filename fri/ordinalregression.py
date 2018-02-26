from sklearn.utils import check_X_y

import fri.bounds
from fri.base import FRIBase
import fri.base

class FRIOrdinalRegression(FRIBase):
    """
        Class for ordinal regression data
    
    """
    # TODO: add Min and max problems
    #minProblem = MinProblemOrdinalRegression
    #maxProblem1 = MaxProblem1OrdinalRegression
    #maxProblem2 = MaxProblem2OrdinalRegression

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: add init. Model
        #self.initModel = L1OrdinalRegressor 

        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}
        # Only use parameter grid when no parameter is given
        
        # TODO: Define Parameters for which need to be set and can be optimized by gridsearch

        # if self.C is None:
        #     self.tuned_parameters["C"] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        # else:
        #     self.tuned_parameters["C"] = [self.C]

        # if self.epsilon is None:
        #     self.tuned_parameters["epsilon"] = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # else:
        #     self.tuned_parameters["epsilon"] = [self.epsilon]
    def fit(self, X, y):
        """ Fit model to data and provide feature relevance intervals
        Parameters
        ----------
        X : array_like
            standardized data matrix
        y : array_like
            response vector
        """
        # TODO: implement correct check for ordinal data
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)

        #super().fit(X, y)