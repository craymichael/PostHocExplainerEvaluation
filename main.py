from pprint import pprint

import numpy as np
from alibi.explainers import KernelShap

# TODO: gSHAP-linear, gSHAP-spline, etc.
from posthoceval.global_shap import GlobalKernelShap
from posthoceval.model_generation import AdditiveModel


def evaluate_shap():
    # Make model

    explainer = KernelShap(
        pred_func,
        feature_names=feature_names,
        task='regression'
    )
    explainer.fit(data)

    # explanation = explainer.explain(data[:100])
    # pprint(explanation)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated' for
    # the aggregated scores over all outputs
    explanation = explainer.explain(data)
    expected_value = explanation.expected_value
    shap_values = explanation.shap_values[0]
    outputs = explanation.raw['raw_prediction']
    shap_values_g = explanation.raw['importances']['0']

    print(expected_value, type(expected_value))

    print('outputs', outputs)

    print('shap_values', shap_values)

    print('shap_values_g', shap_values_g)

    shap_preds = np.sum(shap_values, axis=1) + expected_value
    print('sum of abs local error', (shap_preds - outputs).sum())

    gshap = GlobalKernelShap(data, shap_values, expected_value)
    gshap_preds, gshap_vals = gshap.predict(data, return_shap_values=True)

    print('gshap_preds', gshap_preds)

    print('sum of abs global error', (gshap_preds - outputs).sum())

    print('gshap_vals', gshap_vals)

def main():
    pass


if __name__ == '__main__':
    main()
