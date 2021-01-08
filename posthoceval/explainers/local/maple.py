"""
maple.py - A PostHocExplainerEvaluation file

Directly derived from https://github.com/GDPlumb/MAPLE
Commit 427b6048d088624fa6544203cac84bcca675e287
Paper: https://arxiv.org/abs/1807.02910

Copyright (C) The original authors (see above)
    2021  Modified by Zach Carmichael
"""
from typing import Optional

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from posthoceval.explainers._base import BaseExplainer
from posthoceval.model_generation import AdditiveModel


# TODO: abstract model and seed into BaseExplainer (or other abstract...)
class MAPLEExplainer(BaseExplainer):
    def __init__(self,
                 model: AdditiveModel,
                 train_size: float = 2 / 3,
                 seed: Optional[int] = None,
                 **kwargs):
        # at 7cecf35621859a9ce915da1947a5fb90ee313f08, MAPLE uses 2/3
        # train/val split in Code/Misc.py
        self.train_size = train_size
        self.model = model

        self.seed = seed

        self.explainer_kwargs = kwargs
        self._explainer: Optional[_MAPLE] = None

    def fit(self, X, y=None):
        if y is None:
            y = self.model(X)

        # split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.train_size, random_state=self.seed)

        self._explainer = _MAPLE(
            X_train=X_train,
            MR_train=y_train,
            X_val=X_val,
            MR_val=y_val,
            seed=self.seed,
            **self.explainer_kwargs,
        )

    def predict(self, X):
        if self._explainer is None:
            raise RuntimeError('Must call fit() before predict()')

        return self._explainer.predict(X)

    def feature_contributions(self, X, return_y=False):
        if self._explainer is None:
            raise RuntimeError('Must call fit() before obtaining feature '
                               'contributions')

        contribs_maple = []
        for xi in X:
            coefs = self._explainer.explain(xi)['coefs']
            # coefs[0] is the intercept, throw it in the trash
            contribs_maple.append(
                coefs[1:] * xi
            )

        contribs_maple = np.asarray(contribs_maple)
        return contribs_maple


class _MAPLE:
    """see header of file for attribution. this class has been modified and
    reformatted without changing any functionality"""

    def __init__(self, X_train, MR_train, X_val, MR_val, fe_type='rf',
                 n_estimators=200, max_features=0.5, min_samples_leaf=10,
                 regularization=0.001, seed=None):
        # Features and the target model response
        self.X_train = X_train
        self.MR_train = MR_train
        self.X_val = X_val
        self.MR_val = MR_val

        # Forest Ensemble Parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf

        # Local Linear Model Parameters
        self.regularization = regularization

        # Data parameters
        self.num_features = X_train.shape[1]
        self.num_train = X_train.shape[0]
        num_val = X_val.shape[0]

        self.random_state = np.random.RandomState(seed=seed)

        # Fit a Forest Ensemble to the model response
        if fe_type == 'rf':
            fe = RandomForestRegressor(n_estimators=n_estimators,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features,
                                       random_state=self.random_state)
        elif fe_type == 'gbrt':
            fe = GradientBoostingRegressor(n_estimators=n_estimators,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=max_features,
                                           max_depth=None,
                                           random_state=self.random_state)
        else:
            import sys
            sys.exit(f'Unknown FE type {fe_type}')

        fe.fit(X_train, MR_train)
        self.fe = fe

        train_leaf_ids = fe.apply(X_train)
        self.train_leaf_ids = train_leaf_ids

        val_leaf_ids_list = fe.apply(X_val)

        # Compute the feature importances: Non-normalized @ Root
        scores = np.zeros(self.num_features)
        for i in range(n_estimators):
            # else fe_type == 'gbrt'
            est_idx = i if fe_type == 'rf' else (i, 0)
            # -2 indicates leaf, index 0 is root
            splits = fe[est_idx].tree_.feature
            if splits[0] != -2:
                # impurity reduction not normalized per tree
                scores[splits[0]] += fe[est_idx].tree_.impurity[0]
        self.feature_scores = scores
        mostImpFeats = np.argsort(-scores)

        # Find the number of features to use for MAPLE
        retain_best = 0
        rmse_best = np.inf
        for retain in range(1, self.num_features + 1):

            # Drop less important features for local regression
            X_train_p = np.delete(X_train, mostImpFeats[retain:], axis=1)
            X_val_p = np.delete(X_val, mostImpFeats[retain:], axis=1)

            lr_predictions = np.empty([num_val], dtype=float)

            for i in range(num_val):
                weights = self.training_point_weights(val_leaf_ids_list[i])

                # Local linear model
                lr_model = Ridge(alpha=regularization,
                                 random_state=self.random_state)
                lr_model.fit(X_train_p, MR_train, weights)
                lr_predictions[i] = lr_model.predict(X_val_p[i].reshape(1, -1))

            rmse_curr = np.sqrt(mean_squared_error(lr_predictions, MR_val))

            if rmse_curr < rmse_best:
                rmse_best = rmse_curr
                retain_best = retain

        self.retain = retain_best
        self.X = np.delete(X_train, mostImpFeats[retain_best:], axis=1)

    def training_point_weights(self, instance_leaf_ids):
        weights = np.zeros(self.num_train)
        for i in range(self.n_estimators):
            # Get the PNNs for each tree (ones with the same leaf_id)
            PNNs_Leaf_Node = np.where(
                self.train_leaf_ids[:, i] == instance_leaf_ids[i])
            weights[PNNs_Leaf_Node] += 1.0 / len(PNNs_Leaf_Node[0])
        return weights

    def explain(self, x):
        x = x.reshape(1, -1)

        mostImpFeats = np.argsort(-self.feature_scores)
        x_p = np.delete(x, mostImpFeats[self.retain:], axis=1)

        curr_leaf_ids = self.fe.apply(x)[0]
        weights = self.training_point_weights(curr_leaf_ids)

        # Local linear model
        lr_model = Ridge(alpha=self.regularization,
                         random_state=self.random_state)
        lr_model.fit(self.X, self.MR_train, weights)

        # Get the model coefficients
        coefs = np.zeros(self.num_features + 1)
        coefs[0] = lr_model.intercept_
        coefs[np.sort(mostImpFeats[0:self.retain]) + 1] = lr_model.coef_

        # Get the prediction at this point
        prediction = lr_model.predict(x_p.reshape(1, -1))

        out = {'weights': weights, 'coefs': coefs, 'pred': prediction}

        return out

    def predict(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
        for i in range(n):
            exp = self.explain(X[i, :])
            pred[i] = exp['pred'][0]
        return pred

    # Make the predictions based on the forest ensemble (either random forest
    # or gradient boosted regression tree) instead of MAPLE
    def predict_fe(self, X):
        return self.fe.predict(X)

    # Make the predictions based on SILO (no feature selection) instead of MAPLE
    def predict_silo(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
        # The contents of this inner loop are similar to explain(): doesn't use
        # the features selected by MAPLE or return as much information
        for i in range(n):
            x = X[i, :].reshape(1, -1)

            curr_leaf_ids = self.fe.apply(x)[0]
            weights = self.training_point_weights(curr_leaf_ids)

            # Local linear model
            lr_model = Ridge(alpha=self.regularization,
                             random_state=self.random_state)
            lr_model.fit(self.X_train, self.MR_train, weights)

            pred[i] = lr_model.predict(x)[0]

        return pred
