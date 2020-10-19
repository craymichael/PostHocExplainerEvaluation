from pprint import pprint

import numpy as np
from alibi.explainers import KernelShap

# TODO: gSHAP-linear, gSHAP-spline, etc.
from posthoceval.global_shap import GlobalKernelShap


class DummyModel(object):
    def inference(self, x):
        return np.asarray([*map(self.infer_once, x)])

    def infer_once(self, x):
        x1, x2, x3 = x
        if x1 >= 50:
            return x2 * x3
        elif x2 < 0:
            return x1
        else:
            return x3


def generate_data(n_samples, n_features, a=-100, b=100):
    return np.random.uniform(a, b, size=(n_samples, n_features))


def main():
    model = DummyModel()
    data = generate_data(100_000, 3)

    explainer = KernelShap(
        model.inference,
        feature_names=['x1', 'x2', 'x3']
    )
    explainer.fit(data)

    outputs = model.inference(data)
    print('E[Y] =', outputs.mean())

    # explanation = explainer.explain(data[:100])
    # pprint(explanation)

    test = np.asarray([[175, 2, 2]])
    print('input ', test)
    print('output', model.inference(test))
    explanation = explainer.explain(test)
    pprint(explanation)


if __name__ == '__main__':
    main()
