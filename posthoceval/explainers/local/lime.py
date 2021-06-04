
import logging

from typing import Optional
from typing import Union

import numpy as np

from posthoceval.profile import profile
from posthoceval.explainers._base import BaseExplainer
from posthoceval.models.model import AdditiveModel

logger = logging.getLogger(__name__)


class LIMETabularExplainer(BaseExplainer):
    

    def __init__(self,
                 model: AdditiveModel,
                 seed: Optional[int] = None,
                 task: str = 'regression',
                 num_samples: int = 5000,
                 verbose: Union[int, bool] = 1):
        
        super().__init__(
            model=model,
            tabular=True,
            seed=seed,
            task=task,
            verbose=verbose,
        )

        self.num_samples = num_samples

        self._mean = None
        self._scale = None

    def _fit(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            grouped_feature_names=None,  
    ):
        if self.verbose > 0:
            logger.info('Fitting LIME')

        
        
        
        
        
        
        

        
        from lime.lime_tabular import (
            LimeTabularExplainer as _LimeTabularExplainer)

        self._explainer: Optional[_LimeTabularExplainer]
        self._explainer = _LimeTabularExplainer(
            training_data=X,
            feature_names=range(self.model.n_features),
            mode=self.task,
            random_state=self.seed,
            discretize_continuous=False,
        )

        
        self._mean = self._explainer.scaler.mean_
        self._scale = self._explainer.scaler.scale_

    def predict(self, X):
        return NotImplementedError

    def _process_explanation(self, expl_vals_i, xi, intercept_i=None):
        
        _, coefs_i = zip(*sorted(expl_vals_i, key=lambda x: x[0]))
        
        coefs_i = np.asarray(coefs_i) / self._scale
        
        contribs_i = coefs_i * xi

        
        if intercept_i is not None:
            intercept_i = (intercept_i -
                           np.sum(coefs_i * self._mean))
            return contribs_i, intercept_i
        return contribs_i

    @profile
    def _call_explainer(self, X: np.ndarray):
        
        

        if self._explainer is None:  
            raise RuntimeError('Must call fit() before obtaining feature '
                               'contributions')

        if self.verbose > 0:
            logger.info('Fetching LIME explanations')

        contribs_lime = []
        intercepts = []

        explain_kwargs = {
            'predict_fn': self.model,
            'num_features': X.shape[-1],
            'num_samples': self.num_samples,
        }
        if self.task == 'classification':
            
            
            explain_kwargs['top_labels'] = self.model(X[:1]).shape[-1]

        for i, xi in enumerate(X):
            expl_i = self._explainer.explain_instance(xi, **explain_kwargs)
            expl_i_map = expl_i.as_map()

            if self.task == 'classification':
                contribs_i = []
                intercept_i = []
                for k in range(explain_kwargs['top_labels']):
                    
                    expl_ik = expl_i_map[k]

                    contribs_ik, intercept_ik = self._process_explanation(
                        expl_ik, xi, intercept_i=expl_i.intercept[k])
                    intercept_i.append(intercept_ik)
                    contribs_i.append(contribs_ik)
            else:  
                expl_i1 = expl_i_map[1]

                contribs_i, intercept_i = self._process_explanation(
                    expl_i1, xi, intercept_i=expl_i.intercept[1])

            
            contribs_lime.append(contribs_i)
            intercepts.append(intercept_i)

        contribs_lime = np.asarray(contribs_lime)
        intercepts = np.asarray(intercepts)

        if self.task == 'classification':
            
            contribs_lime = np.moveaxis(contribs_lime, 0, 1)
            
            intercepts = intercepts.T
            
            y_expl = contribs_lime.sum(axis=2) + intercepts
        else:
            
            y_expl = contribs_lime.sum(axis=1) + intercepts

        return {'contribs': contribs_lime, 'intercepts': intercepts,
                'predictions': y_expl}


LIMEExplainer = LIMETabularExplainer
