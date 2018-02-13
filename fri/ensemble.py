import math
from multiprocessing.pool import Pool

import numpy as np
from sklearn import clone
from sklearn.utils import check_random_state, resample

from fri.base import FRIBase
from fri.classification import  FRIClassification

class EnsembleFRI(FRIBase):
    def __init__(self, model, n_bootstraps=10, random_state=None, n_jobs=1):
        self.random_state = check_random_state(random_state)
        self.n_bootstraps = n_bootstraps
        self.model = model
        self.n_jobs = n_jobs
        self.bs_seed = self.random_state.randint(10000000)  # Seed for bootstraps rngs, constant for multiprocessing

        if isinstance(self.model, FRIClassification):
            isRegression = False
        else:
            isRegression = True

        model._ensemble = True
        model.random_state = self.random_state
        super().__init__(isRegression, random_state=self.random_state)

    def _fit_one_bootstrap(self, i):
        m = clone(self.model)
        m._ensemble = True

        X, y = self.X_, self.y_
        n = X.shape[0]
        n_samples = math.ceil(0.8 * n)

        # Get bootstrap set
        X_bs, y_bs = resample(X, y, replace=True,
                              n_samples=n_samples,
                              random_state=self.bs_seed + i)

        m.fit(X_bs, y_bs)

        if self.model.shadow_features:
            return m.interval_, m._omegas, m._biase, m._shadowintervals
        else:
            return m.interval_, m._omegas, m._biase

    def fit(self, X, y):

        if isinstance(self.model, FRIClassification):
            self.isRegression = False
        else:
            self.isRegression = True

        # Switch for parallel processing, defines map function
        if self.n_jobs > 1:
            def pmap(*args):
                with Pool(self.n_jobs) as p:
                    return p.map(*args)

            nmap = pmap
        else:
            nmap = map

        # Save data for worker functions
        y = np.asarray(y)
        self.X_, self.y_ = X, y

        # run bootstrap iterations
        results = list(nmap(self._fit_one_bootstrap, range(self.n_bootstraps)))

        # Claim result data from workers depending
        if self.model.shadow_features:
            rangevector, omegas, biase, shadowrangevector = zip(*results)
            self._shadowintervals = np.mean(shadowrangevector, axis=0)
        else:
            rangevector, omegas, biase = zip(*results)

        # Aggregation step - we use a simple average
        # Get average
        self.interval_ = np.mean(rangevector, axis=0)
        self._omegas = np.mean(omegas, axis=0)
        self._biase = np.mean(biase, axis=0)

        # Classify features
        self.model._initEstimator(X, y)
        self._svm_clf = self.model._svm_clf
        self._get_relevance_mask()

        return self

    def score(self, X, y):
        return self.model.score(X, y)