import scipy
from sklearn import preprocessing
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from fri.base import FRIBase
from fri.l1models import L1HingeHyperplane
from fri.optproblems import BaseClassificationProblem


class FRIClassification(FRIBase):
    """Class for performing FRI on classification data.
    
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
    problemType = BaseClassificationProblem

    def __init__(self, C=None, optimum_deviation=0.001,
                 random_state=None,
                 n_jobs=None, n_resampling=40, iter_psearch=50, verbose=0):
        super().__init__(C=C, random_state=random_state,
                         n_jobs=n_jobs,
                         n_resampling=n_resampling,iter_psearch=iter_psearch,
                         verbose=verbose, optimum_deviation=optimum_deviation)

    def fit(self, X, y):
        """ Used for fitting the model on the data.

        Parameters
        ----------
        X : array_like
            standardized data matrix
        y : array_like
            label vector

        Raises
        ------
        ValueError
            Only binary classification.
        """

        self.initModel = L1HingeHyperplane

        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}
        # Only use parameter grid when no parameter is given
        if self.C is None:
            self.tuned_parameters["C"] = scipy.stats.reciprocal(a=1e-3, b=1e3)
            # self.tuned_parameters["C"] = np.logspace(-5, 2, self.iter_psearch)
        else:
            self.tuned_parameters["C"] = [self.C]

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        if len(self.classes_) > 2:
            raise ValueError("Only binary class data supported")
        # Negative class is set to -1 for decision surface
        y = preprocessing.LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        super().fit(X, y)