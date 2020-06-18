import scipy
import umap
import hdbscan

import warnings
from abc import abstractmethod

import numpy as np
import math

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer
from sklearn.externals.joblib import Parallel,delayed

from .utils import distance
from .bounds import LowerBound, UpperBound, ShadowLowerBound, ShadowUpperBound
from .l1models import L1OrdinalRegressor, ordinal_scores, L1HingeHyperplane


    def _run_with_single_dim_single_value_preset(self,i, preset_i, n_tries=10):
            """
            Method to run method once for one restricted feature
            Parameters
            ----------
            i:
                restricted feature
            preset_i:
                restricted range of feature i (set before optimization = preset)
            n_tries:
                number of allowed relaxation steps for the L1 constraint in case of LP infeasible

            """
            X = self.X_
            y = self.y_
            # Do we have intervals?
            check_is_fitted(self, "interval_")
            interval = self.unmod_interval_
            d = len(interval)

            constrained_ranges_diff = np.zeros((d, 2))

            # Init empty preset
            preset = np.empty(shape=(d, 2))
            preset.fill(np.nan)

            # Add correct sign of this coef
            signed_preset_i = np.sign(self._svm_coef[0][i]) * preset_i
            preset[i] = signed_preset_i
            # Calculate all bounds with feature i set to min_i
            l1 = self.optim_L1_
            loss = self.optim_loss_
            for j in range(n_tries):
                # try several times if problem to stringent
                try:
                    kwargs = {"verbose": False, "solver": "ECOS"}
                    rangevector, _, _, _ = self._main_opt(X, y, loss,
                                                          l1,
                                                          self.random_state,
                                                          False, presetModel=preset,
                                                          solverargs=kwargs)
                except NotFeasibleForParameters:
                    preset[i] *= -1
                    #print("Community detection: Constrained run failed, swap sign".format)
                    continue
                else:
                    #print("solved constrained opt for ", i)
                    # problem was solvable
                    break
            else:
                raise NotFeasibleForParameters("Grouping failed.", "dim {}".format(i))

            # rangevector, _ = self._postprocessing(self.optim_L1_, rangevector, False,
            #                                      None)
            # Get differences for constrained intervals to normal intervals
            constrained_ranges_diff = self.unmod_interval_ - rangevector

            # Current dimension is not constrained, so these values are set accordingly
            rangevector[i] = preset_i
            constrained_ranges_diff[i] = 0

            return rangevector, constrained_ranges_diff

    def constrained_intervals_(self, preset):
        """Method to return relevance intervals which are constrained using preset ranges or values.
        
        Parameters
        ----------
        preset : array like [[preset lower_Bound_0,preset upper_Bound_0],...,]
            An array where all entries which are not 'np.nan' are interpreted as constraint for that corresponding feature.
            
            Best created using 

            >>> np.full_like(fri_model.interval_, np.nan, dtype=np.double)

            Example: To set  feature 0 to a fixed value use 

            >>> preset[0] = fri_model.interval_[0, 0]
        
        Returns
        -------
        array like
            Relevance bounds with user constraints 
        """
        processed = preset*self.optim_L1_ # Revert scaling to L1 norm which is done for our output intervals (see postprocessing)
        return self._run_with_multiple_value_preset(preset=processed)

    def _run_with_multiple_value_preset(self, preset=None):
            """
            Method to run method with preset values
            """
            X = self.X_
            y = self.y_
            # Do we have intervals?
            check_is_fitted(self, "interval_")
            interval = self.unmod_interval_
            d = len(interval)

            constrained_ranges_diff = np.zeros((d, 2))

            # Add correct sign of this coef
            signed_presets = np.sign(self._svm_coef[0]) * preset.T
            signed_presets = signed_presets.T
            # Calculate all bounds with feature presets
            l1 = self.optim_L1_
            loss = self.optim_loss_
            sumofpreset = np.nansum(preset[:,1])
            if sumofpreset > l1:
                print("maximum L1 norm of presets: ",sumofpreset)
                print("L1 allowed:",l1)
                print("Presets are not feasible. Try lowering values.")
                return
            try:
                kwargs = {"verbose": False, "solver": "ECOS"}
                rangevector, _, _, _ = self._main_opt(X, y, loss,
                                                      l1,
                                                      self.random_state,
                                                      False, presetModel=signed_presets,
                                                      solverargs=kwargs)
            except NotFeasibleForParameters:
                print("Presets are not feasible")
                return


            constrained_ranges_diff = self.unmod_interval_ - rangevector

            # Current dimension is not constrained, so these values are set accordingly
            for i, p in enumerate(preset):
                if np.all(np.isnan(p)):
                    continue
                else:
                    rangevector[i] = p
            rangevector, _ = self._postprocessing(self.optim_L1_, rangevector, False,
                                                              None)
            return rangevector

    def grouping(self, cutoff_threshold=0.55, method="single"):
        """ Find feature clusters based on observed variance when changing feature contributions

        Parameters
        ----------
        cutoff_threshold : float, optional
            Cutoff value for the flat clustering step; decides at which height in the dendrogram the cut is made to determine groups.
        method : str, optional
            Linkage method used in the hierarchical clustering.

        Returns
        -------
        self
        """

        # Do we have intervals?
        check_is_fitted(self, "interval_")
        interval = self.unmod_interval_
        d = len(interval)

        # Init arrays
        interval_constrained_to_min = np.zeros(
            (d, d, 2))  # Save ranges (d,2-dim) for every contrained run (d-times)
        absolute_delta_bounds_summed_min = np.zeros((d, d, 2))
        interval_constrained_to_max = np.zeros(
            (d, d, 2))  # Save ranges (d,2-dim) for every contrained run (d-times)
        absolute_delta_bounds_summed_max = np.zeros((d, d, 2))

        # Set weight for each dimension to minimum and maximum possible value and run optimization of all others
        # We retrieve the relevance bounds and calculate the absolute difference between them and non-constrained bounds
        for i in range(d):
            # min
            ranges, diff = self._run_with_single_dim_single_value_preset(i, interval[i, 0])
            interval_constrained_to_min[i] = ranges
            absolute_delta_bounds_summed_min[i] = diff
            # max
            ranges, diff = self._run_with_single_dim_single_value_preset(i, interval[i, 1])
            interval_constrained_to_max[i] = ranges
            absolute_delta_bounds_summed_max[i] = diff

        feature_points = np.zeros((d, 2 * d * 2))
        for i in range(d):
            feature_points[i, :(2 * d)] = absolute_delta_bounds_summed_min[i].flatten()
            feature_points[i, (2 * d):] = absolute_delta_bounds_summed_max[i].flatten()

        self.relevance_variance = feature_points

        # Calculate similarity using custom measure
        dist_mat = scipy.spatial.distance.pdist(feature_points, metric=distance)

        # Single Linkage clustering
        # link = linkage(dist_mat, method="single")

        link = linkage(dist_mat, method=method, optimal_ordering=True)

        # Set cutoff at which threshold the linkage gets flattened (clustering)
        RATIO = cutoff_threshold
        threshold = RATIO * np.max(link[:, 2])  # max of branch lengths (distances)
        feature_clustering = fcluster(link, threshold, criterion="distance")

        self.feature_clusters_, self.linkage_ = feature_clustering, link

        return self.feature_clusters_

    def umap(self, n_neighbors=2, n_components=2, min_dist=0.1):
        if self.relevance_variance is None:
            print("Use grouping() first to compute relevance_variance")
            return

        um = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                       min_dist=min_dist, metric=distance)
        embedding = um.fit_transform(self.relevance_variance)
        self.relevance_var_embedding_ = embedding
        return embedding

    def grouping_umap(self, only_relevant=False,
                      min_group_size = 2,umap_n_neighbors=2,umap_n_components=2, umap_min_dist=0.1):

        self._umap_embedding = self.umap(n_neighbors=umap_n_neighbors, n_components=umap_n_components,min_dist=umap_min_dist)

        if only_relevant:
            embedding = self._umap_embedding[self.allrel_prediction_]
        else:
            embedding = self._umap_embedding

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_group_size)
        hdb.fit(embedding)
        labels = np.full_like(self.allrel_prediction_,-2,dtype=int)

        if only_relevant:
            labels[self.allrel_prediction_] = hdb.labels_
        else:
            labels = hdb.labels_

        self.group_labels_ = labels

        return labels

    def grouping_hdbscan(self, only_relevant=False, min_group_size = 2):

        if only_relevant:
            data = self.relevance_variance[self.allrel_prediction_]
        else:
            data = self.relevance_variance

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_group_size,metric=distance
            )
        hdb.fit(data)
        labels = np.full_like(self.allrel_prediction_,-2,dtype=int)

        if only_relevant:
            labels[self.allrel_prediction_] = hdb.labels_
        else:
            labels = hdb.labels_

        self.group_labels_ = labels

        return labels
