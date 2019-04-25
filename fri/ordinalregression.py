import numpy as np
import scipy
from sklearn.utils import check_X_y

from fri.base import FRIBase
from fri.l1models import L1OrdinalRegressor
from fri.optproblems import BaseOrdinalRegressionProblem


class FRIOrdinalRegression(FRIBase):
    """Class for performing FRI on ordinal regression data.
    
    Parameters
    ----------
    C : float , optional
        Regularization parameter, default obtains the hyperparameter through gridsearch optimizing accuracy
    random_state : object
        Set seed for random number generation.
    n_resampling : integer ( Default = 40)
        Number of probe feature permutations used. 
    iter_psearch : integer ( Default = 50)
        Amount of samples used for parameter search.
        Trade off between finer tuned model performance and run time of parameter search.
    n_jobs : int, optional
        Enables parallel computation (>1) of feature intervals.
        If -1, use all cores.
    optimum_deviation : float, optional (Default = 0.001)
        Rate of allowed deviation from the optimal solution (L1 norm of model weights).
        Default allows one percent deviation. 
        Allows for more relaxed optimization problems and leads to bigger intervals which are easier to interpret.
        Setting to 0 allows the best feature selection accuracy.
    verbose : int ( Default = 0)
        Print out verbose messages. The higher the number, the more messages are printed.
    
    Attributes
    ----------
    allrel_prediction_ : array of booleans
        Truth value for each feature if it is relevant (weakly OR strongly).
    interval_ : array [[lower_Bound_0,UpperBound_0],...,]
        Relevance bounds in 2D array format.
    optim_L1_ : double
        L1 norm of baseline model.
    optim_loss_ : double
        Sum of slack (loss) of baseline model.
    optim_model_ : fri.l1models object
        Baseline model
    optim_score_ : double
        Score of baseline model
    relevance_classes_ : array like
        Array with classification of feature relevances: 2 denotes strongly relevant, 1 weakly relevant and 0 irrelevant.
    unmod_interval_ : array like
        Same as `interval_` but not scaled to L1.
    
    """

    problemType = BaseOrdinalRegressionProblem

    def __init__(self, C=None, optimum_deviation=0.001, random_state=None,
                 n_jobs=None, n_resampling=40, iter_psearch=50, verbose=0, **kwargs):
        super().__init__(C=C, random_state=random_state,
                         n_jobs=n_jobs,
                         n_resampling=n_resampling,iter_psearch=iter_psearch,
                         verbose=verbose, optimum_deviation=optimum_deviation)

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

        self.initModel = L1OrdinalRegressor

        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}

        # Only use parameter grid when no parameter is given
        if self.C is None:
            self.tuned_parameters["C"] = scipy.stats.reciprocal(a=1e-7, b=1e7)
            # self.tuned_parameters["C"] = np.logspace(-5, 2, self.iter_psearch)
        else:
            self.tuned_parameters["C"] = [self.C]

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        if np.min(y) > 0:
            print("First ordinal class has index > 0. Trying to shift indices.")
            y = y - np.min(y)

        # Get ordinal classes
        self.classes_ = np.unique(y)

        super().fit(X, y)