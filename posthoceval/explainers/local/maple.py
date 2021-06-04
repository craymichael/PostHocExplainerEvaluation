
from typing import Optional
from typing import List
from typing import Union

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel
from posthoceval.rand import as_random_state
from posthoceval.rand import randint


class _MAPLE:

    def __init__(self, X_train, MR_train, X_val, MR_val, fe_type='rf',
                 n_estimators=200, max_features=0.5, min_samples_leaf=10,
                 regularization=0.001, seed=None):
         
        self.X_train = X_train
        self.MR_train = MR_train
        self.X_val = X_val
        self.MR_val = MR_val

         
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf

         
        self.regularization = regularization

         
        self.num_features = X_train.shape[1]
        self.num_train = X_train.shape[0]
        num_val = X_val.shape[0]

        self.random_state = np.random.RandomState(seed=seed)

         
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

         
        scores = np.zeros(self.num_features)
        for i in range(n_estimators):
             
            est_idx = i if fe_type == 'rf' else (i, 0)
             
            splits = fe[est_idx].tree_.feature
            if splits[0] != -2:
                 
                scores[splits[0]] += fe[est_idx].tree_.impurity[0]
        self.feature_scores = scores
        mostImpFeats = np.argsort(-scores)

         
        retain_best = 0
        rmse_best = np.inf
        for retain in range(1, self.num_features + 1):

             
            X_train_p = np.delete(X_train, mostImpFeats[retain:], axis=1)
            X_val_p = np.delete(X_val, mostImpFeats[retain:], axis=1)

            lr_predictions = np.empty([num_val], dtype=float)

            for i in range(num_val):
                weights = self.training_point_weights(val_leaf_ids_list[i])

                 
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

         
        lr_model = Ridge(alpha=self.regularization,
                         random_state=self.random_state)
        lr_model.fit(self.X, self.MR_train, weights)

         
        coefs = np.zeros(self.num_features + 1)
        coefs[0] = lr_model.intercept_
        coefs[np.sort(mostImpFeats[:self.retain]) + 1] = lr_model.coef_

         
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

     
     
    def predict_fe(self, X):
        return self.fe.predict(X)

     
    def predict_silo(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
         
         
        for i in range(n):
            x = X[i, :].reshape(1, -1)

            curr_leaf_ids = self.fe.apply(x)[0]
            weights = self.training_point_weights(curr_leaf_ids)

             
            lr_model = Ridge(alpha=self.regularization,
                             random_state=self.random_state)
            lr_model.fit(self.X_train, self.MR_train, weights)

            pred[i] = lr_model.predict(x)[0]

        return pred


class MAPLEExplainer(BaseExplainer):
    _explainer: Optional[Union[List[_MAPLE], _MAPLE]]

    def __init__(self,
                 model: AdditiveModel,
                 train_size: float = 2 / 3,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 max_samples: int = 10000,
                 **kwargs):
        super().__init__(
            model=model,
            tabular=True,
            seed=seed,
            task=task,
            verbose=False,
        )

         
         
        self.train_size = train_size
        self.max_samples = max_samples
        self.explainer_kwargs = kwargs

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,
    ):
         
        n_samples = round(self.max_samples * 25 / X.shape[1])
        sample_idxs = None
        if len(X) > n_samples:
            rng = as_random_state(self.seed)
            sample_idxs = randint(0, len(X), size=n_samples, seed=rng)
            X = X[sample_idxs]

        if y is None:
            y = self.model(X)
        elif sample_idxs is not None:
            y = y[sample_idxs]

        if self.task == 'regression' and y.ndim == 2:
            y = np.squeeze(y, axis=1)
        else:
            y = np.reshape(y, (len(y), -1))

         
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.train_size, random_state=self.seed)

        if self.task == 'regression':
            self._explainer = _MAPLE(
                X_train=X_train,
                MR_train=y_train,
                X_val=X_val,
                MR_val=y_val,
                seed=self.seed,
                **self.explainer_kwargs,
            )
        else:
            self._explainer = [_MAPLE(
                X_train=X_train,
                MR_train=y_train[:, k],
                X_val=X_val,
                MR_val=y_val[:, k],
                seed=self.seed,
                **self.explainer_kwargs,
            ) for k in range(y.shape[1])]

    def predict(self, X):
        if self._explainer is None:
            raise RuntimeError('Must call fit() before predict()')

        if self.task == 'regression':
            return self._explainer.predict(X)
        else:
            raise NotImplementedError   

    def _call_explainer(self, X):
        if self._explainer is None:
            raise RuntimeError('Must call fit() before obtaining feature '
                               'contributions')

        contribs_maple = []
        intercepts = []
        y_maple = []
        for xi in X:
            if self.task == 'regression':
                contribs_i, intercepts_i, pred_i = (
                    self._call_explainer_one_class(self._explainer, xi))
            else:
                contribs_i = []
                intercepts_i = []
                pred_i = []
                for explainer in self._explainer:
                    contrib, intercept, pred = self._call_explainer_one_class(
                        explainer, xi)
                    contribs_i.append(contrib)
                    intercepts_i.append(intercept)
                    pred_i.append(pred)
            contribs_maple.append(contribs_i)
            intercepts.append(intercepts_i)
            y_maple.append(pred_i)

        contribs_maple = np.asarray(contribs_maple)
        if self.task == 'classification':
            contribs_maple = np.moveaxis(contribs_maple, 0, 1)

        y_maple = np.stack(y_maple).squeeze(axis=-1)
        if self.task == 'classification':
            y_maple = y_maple.T

        return {'contribs': contribs_maple, 'intercepts': intercepts,
                'predictions': y_maple}

    @staticmethod
    def _call_explainer_one_class(explainer, xi):
        explanation = explainer.explain(xi)
        coefs = explanation['coefs']
        contrib = coefs[1:] * xi
        intercept = coefs[0]
        pred = explanation['pred']
        return contrib, intercept, pred
