from sklearn import preprocessing
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from fri.base import FRIBase
from fri.l1models import L1HingeHyperplane
from fri.optproblems import MinProblemClassification, MaxProblem1, MaxProblem2


class FRIClassification(FRIBase):
    minProblem = MinProblemClassification
    maxProblem1 = MaxProblem1
    maxProblem2 = MaxProblem2

    def __init__(self, C=None, optimum_deviation=0.01,
                 random_state=None,shadow_features=False,
                 parallel=False, n_resampling=3,
                 feat_elim=False, debug=False):
        super().__init__(isRegression=False, C=C, random_state=random_state,
                         shadow_features=shadow_features, parallel=parallel,
                         feat_elim=feat_elim,n_resampling=n_resampling,
                         debug=debug, optimum_deviation=optimum_deviation)
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