from pprint import pprint

import numpy as np

from alibi.explainers import KernelShap
from alibi.explainers.shap_wrappers import KERNEL_SHAP_BACKGROUND_THRESHOLD

from posthoceval.global_shap import GlobalKernelShap
from posthoceval.model_generation import AdditiveModel
from posthoceval.model_generation import tsang_iclr18_models


def evaluate_shap(debug=False):
    # TODO: gSHAP-linear, gSHAP-spline, etc.

    # Make model
    model = tsang_iclr18_models('f2')

    # Make data
    n_samples = 30_000 if not debug else 100
    # TODO: better data ranges based on continuity of function
    # data = np.random.uniform(-1, +1, size=(n_samples, model.n_features))
    data = np.random.uniform(0, +1, size=(n_samples, model.n_features))

    explainer = KernelShap(
        model,
        feature_names=model.symbol_names,
        task='regression',
    )
    fit_kwargs = {}
    if KERNEL_SHAP_BACKGROUND_THRESHOLD < n_samples:
        fit_kwargs['summarise_background'] = True

    explainer.fit(data, **fit_kwargs)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
    # for the aggregated scores over all outputs
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


if __name__ == '__main__':
    def main():
        import argparse

        parser = argparse.ArgumentParser(
            description=''
        )
        parser.add_argument('--debug', '-B', action='store_true',
                            help='Use fewer samples to make sure things work.')
        args = parser.parse_args()

        evaluate_shap(debug=args.debug)


    main()
