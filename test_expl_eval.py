"""
test_expl_eval.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from posthoceval.expl_eval.metrics import sensitivity_n
from posthoceval.explainers.local.lime import LIMEExplainer
from posthoceval.model_generation import AdditiveModel


def test_sensitivity_n():
    print('Create model')
    model = AdditiveModel.from_expr(
        # '2 * x1 - x2 + x3 * x4 + x5 + x6 + x7 * x8'
        '+'.join(f'x{i}' for i in range(1000))
    )
    print('Generate data')
    X = np.random.rand(100, model.n_features)

    print('Create and fit explainer')
    explainer = LIMEExplainer(
        model
    )
    explainer.fit(X)

    print('Sensitivity N')
    ns, pccs = sensitivity_n(
        model=model,
        explain_func=explainer.feature_contributions,
        X=X,
    )

    import matplotlib.pyplot as plt
    plt.scatter(ns, pccs)
    plt.xlabel('n')
    plt.ylabel('PCC')
    plt.show()


if __name__ == '__main__':
    test_sensitivity_n()
