"""
test_expl_eval.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np

from posthoceval.expl_eval.metrics import sensitivity_n
from posthoceval.expl_eval.metrics import faithfulness_melis
from posthoceval.explainers.local.lime import LIMEExplainer
from posthoceval.model_generation import AdditiveModel


def test_sensitivity_n():
    np.seterr(all='raise')

    print('Create model')
    model = AdditiveModel.from_expr(
        # '2 * x1 - x2 + x3 * x4 + x5 + x6 + x7 * x8'
        '+'.join(f'x{i}' for i in range(64))
    )
    print('Generate data')
    X = np.random.rand(256, model.n_features)

    print('Create and fit explainer')
    explainer = LIMEExplainer(
        model,
        num_samples=500,
    )
    explainer.fit(X)

    print('Explaining')
    attribs = explainer.feature_contributions(X)

    print('Sensitivity N')
    ns, pccs = sensitivity_n(
        model=model,
        attribs=attribs,
        X=X,
        verbose=True,
        aggregate=False,
    )

    print(faithfulness_melis(
        model=model,
        attribs=attribs,
        X=X,
        verbose=True,
    ))

    import matplotlib.pyplot as plt
    print(ns, pccs)
    plt.scatter(ns, pccs)
    plt.xlabel('n')
    plt.ylabel('PCC')
    if plt.get_backend() == 'agg':
        print("plt.savefig('test_expl_eval.png')")
        plt.savefig('test_expl_eval.png')
    else:
        plt.show()


if __name__ == '__main__':
    test_sensitivity_n()
