from sklearn import preprocessing
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

import fri.bounds
from fri.base import FRIBase
from fri.l1models import L1HingeHyperplane
import fri.base

class FRIClassification(FRIBase):
    """
    Class for Classification data
    """

    LowerBound = fri.bounds.LowerBound
    UpperBound = fri.bounds.UpperBound
    LowerBoundS = fri.bounds.ShadowLowerBound
    UpperBoundS = fri.bounds.ShadowUpperBound

    def __init__(self, C=None, random_state=None,
                 shadow_features=False, parallel=False, n_resampling=3, feat_elim=False, **kwargs):
        """Initialize a solver for classification data
        Parameters
        ----------
        C : float , optional
            Regularization parameter, default obtains the hyperparameter
            through gridsearch optimizing accuracy
        random_state : object
            Set seed for random number generation.
        shadow_features : boolean, optional
            Enables noise reduction using feature permutation results.
        parallel : boolean, optional
            Enables parallel computation of feature intervals
        """
        super().__init__(isRegression=False, C=C, random_state=random_state,
                         shadow_features=shadow_features, parallel=parallel, feat_elim=False, **kwargs)
        
        self.initModel = L1HingeHyperplane

        # Define parameters which are optimized in the initial gridsearch
        self.tuned_parameters = {}
        # Only use parameter grid when no parameter is given
        if self.C is None:
            self.tuned_parameters["C"] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        else:
            self.tuned_parameters["C"] = [self.C]

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
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