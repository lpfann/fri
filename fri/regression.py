from sklearn.utils import check_X_y

import fri.bounds
from fri.base import FRIBase
import fri.base

class FRIRegression(FRIBase):
    """
        Class for regression data
    
    """
    LowerBound = fri.bounds.LowerBound
    UpperBound = fri.bounds.UpperBound
    LowerBoundS = fri.bounds.ShadowLowerBound
    UpperBoundS = fri.bounds.ShadowUpperBound

    def __init__(self, epsilon=None, C=None, random_state=None,
                 shadow_features=False, parallel=False, feat_elim=False, **kwargs):
        super().__init__(isRegression=True, C=C, random_state=random_state,
                         shadow_features=shadow_features, parallel=parallel, feat_elim=False, **kwargs)
        self.epsilon = epsilon

    def fit(self, X, y):
        """ Fit model to data and provide feature relevance intervals
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