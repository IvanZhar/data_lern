import numpy as np
import pandas as pd


class BTree:
    def __init__(self, feature_idx=None, min_size=1, max_depth=float("inf"), metric='mae', predict_by='mean',
                 recur_level=0):
        self.right = None
        self.left = None
        self.is_leaf = True
        self.splitter = None
        self.prediction = None
        self.min_size = min_size
        self.max_depth = max_depth
        self.metric = metric.strip().lower()
        self.predict_by = predict_by
        self.feature_idx = feature_idx
        self.recur_level = recur_level

    @staticmethod
    def _mse(y):
        return np.mean((y - np.mean(y)) ** 2) if y.shape[0] != 0 else float("inf")

    @staticmethod
    def _mae(y):
        return np.mean(np.abs(y - np.median(y))) if y.shape[0] != 0 else float("inf")

    def make_split(self, feature_vector, response_vector, metric_full: float):
        
        assert response_vector.shape[0] == feature_vector.shape[0], 'X shape doesnt match Y shape!'
        resp_full_metric_full = response_vector.shape[0] * metric_full

        def qual_eval(splitter):
            nonlocal feature_vector, metric_full, response_vector
            left = response_vector[feature_vector < splitter]
            if len(left) < self.min_size:
                return np.inf
            right = response_vector[feature_vector >= splitter]
            if len(right) < self.min_size:
                return np.inf
            metric_left = self._mae(left) if self.metric == 'mae' else self._mse(left)
            metric_right = self._mae(right) if self.metric == 'mae' else self._mse(right)
            return resp_full_metric_full + (len(left) * metric_left) + (len(right) * metric_right)

        uniq_feat = np.unique(feature_vector)[1:]
        if not len(uniq_feat):
            return None, np.inf
        qual_eval_vec = np.vectorize(qual_eval)
        qual_matrix = qual_eval_vec(uniq_feat)
        best_qual_idx = np.argmin(qual_matrix)
        return uniq_feat[best_qual_idx] if qual_matrix[best_qual_idx] != np.inf else None, np.min(qual_matrix)

    def choose_split(self, response_array, features_array):
        metric_full = self._mae(response_array) if self.metric == 'mae' else self._mse(response_array)

        if features_array.size:
            feature_metric_array = np.apply_along_axis(self.make_split, 0, features_array, response_array,
                                                       metric_full).transpose()

            best_split_feature_idx = np.argmin(feature_metric_array[:, 1])
            best_splitter = feature_metric_array[best_split_feature_idx, 0]
            return best_split_feature_idx, best_splitter
        else:
            return None, None

    def fit(self, response_array, features_array):
        assert response_array.shape[0] == features_array.shape[0], 'X shape doesnt match Y shape'
        if type(response_array) == pd.core.frame.DataFrame or type(response_array) == pd.core.series.Series:
            response_array = response_array.to_numpy()
        if type(features_array) == pd.core.frame.DataFrame or type(features_array) == pd.core.series.Series:
            features_array = features_array.to_numpy()

        self.feature_idx, self.splitter = self.choose_split(response_array, features_array)

        if (self.left is None and self.right is not None) or (self.right is None and self.left is not None):
            exit('Root has just one branch!')

        if self.splitter is not None and self.recur_level < self.max_depth:
            left_response = response_array[features_array[:, self.feature_idx] < self.splitter]
            left_features = features_array[features_array[:, self.feature_idx] < self.splitter]
            right_response = response_array[features_array[:, self.feature_idx] >= self.splitter]
            right_features = features_array[features_array[:, self.feature_idx] >= self.splitter]
            if self.left is None and self.right is None:
                self.left = BTree(min_size=self.min_size, max_depth=self.max_depth, metric=self.metric,
                                  predict_by=self.predict_by, recur_level=self.recur_level + 1)
                self.right = BTree(min_size=self.min_size, max_depth=self.max_depth, metric=self.metric,
                                   predict_by=self.predict_by, recur_level=self.recur_level + 1)
                self.is_leaf = False
            self.left.fit(left_response, left_features)
            self.right.fit(right_response, right_features)
        else:
            if self.predict_by == 'mean':
                self.prediction = np.mean(response_array)
            if self.predict_by == 'median':
                self.prediction = np.median(response_array)
            if self.predict_by == 'mode':
                self.prediction = np.mode(response_array)

    def predict(self, features):
        features_copy = pd.DataFrame(data=np.full((features.shape[0]), np.nan), columns=['Y_predicted'],
                                     index=features.index)
        self.recurs_search(features, features_copy)
        return features_copy

    def recurs_search(self, features, features_copy):
        if not self.is_leaf:
            left_predicted = features.loc[features.iloc[:, self.feature_idx] < self.splitter]
            right_predicted = features.loc[features.iloc[:, self.feature_idx] >= self.splitter]
            if left_predicted.shape[0] != 0:
                self.left.recurs_search(left_predicted, features_copy)
            if right_predicted.shape[0] != 0:
                self.right.recurs_search(right_predicted, features_copy)
        else:
            features_copy.loc[features.index] = self.prediction

