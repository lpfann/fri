from fri.base import FRIBase
from fri.l1models import L1EpsilonRegressor
from fri.optproblems import BaseRegressionProblem
from sklearn.utils import check_X_y


class FRIRegression(FRIBase):
    """Class for performing FRI on regression data.
    
    Parameters
    ----------
    C : float , optional
        Regularization parameter, default obtains the hyperparameter through gridsearch optimizing accuracy
    epsilon: float, optional
        Parameter to determine epsilon region around hyperplane. Related to C.
    random_state : object
        Set seed for random number generation.
    n_resampling : integer ( Default = 3)
        Number of probe feature permutations used. 
    parallel : boolean, optional
        Enables parallel computation of feature intervals
    optimum_deviation : float, optional (Default = 0.001)
        Rate of allowed deviation from the optimal solution (L1 norm of model weights).
        Default allows one percent deviation. 
        Allows for more relaxed optimization problems and leads to bigger intervals which are easier to interpret.
        Setting to 0 allows the best feature selection accuracy.
    debug : boolean
        Enable output of internal values for debugging purposes.
    
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
    tuned_C_ : double
        Chosen reguralisation parameter using cv-gridsearch.
    tuned_epsilon_ : double
        Epsilon parameter for regression baseline model chosen by cv-gridsearch.
    unmod_interval_ : array like
        Same as `interval_` but not scaled to L1.
    
    """
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
        self.classes_ = None

        super().fit(X, y)