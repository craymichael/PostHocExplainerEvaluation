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
    print('Model')
    # model = tsang_iclr18_models('f2')  # SHIT BREAKS
    model = tsang_iclr18_models('f9')
    print('Model:')
    model.pprint()

    # Make data
    print('Data')
    # n_samples = 30_000 if not debug else 10
    n_samples = 300 if not debug else 10
    # TODO: better data ranges based on continuity of function
    # TODO: use valid_variable_domains when properly+cleanly integrated
    # data = np.random.uniform(-1, +1, size=(n_samples, model.n_features))
    data = np.random.uniform(0, +1, size=(n_samples, model.n_features))
    data = data.astype('float32')  # Needed for Theano GPU accel...

    # Split
    train_split_pct = 2 / 3
    split_idx = int(train_split_pct * len(data))
    data_train = data[:split_idx]
    data_test = data[split_idx:]

    # Explainer
    print('Explainer')
    explainer = KernelShap(
        model,
        feature_names=model.symbol_names,
        task='regression',
    )
    fit_kwargs = {}
    if KERNEL_SHAP_BACKGROUND_THRESHOLD < n_samples:
        fit_kwargs['summarise_background'] = True

    print('Explainer fit')
    explainer.fit(data_train, **fit_kwargs)

    # Note: explanation.raw['importances'] has aggregated scores per output with
    # corresponding keys, e.g., '0' & '1' for two outputs. Also has 'aggregated'
    # for the aggregated scores over all outputs
    print('Explain')
    explanation = explainer.explain(data_train)
    expected_value = explanation.expected_value
    shap_values = explanation.shap_values[0]
    outputs_train = explanation.raw['raw_prediction']
    shap_values_g = explanation.raw['importances']['0']

    print('expected_value', expected_value)

    # print('outputs', outputs)
    # print('shap_values', shap_values)
    # print('shap_values_g', shap_values_g)

    shap_preds = np.sum(shap_values, axis=1) + expected_value
    print('sum of abs local error', (shap_preds - outputs_train).sum())

    print('Global SHAP')
    gshap = GlobalKernelShap(data_train, shap_values, expected_value)

    gshap_preds, gshap_vals = gshap.predict(data_train, return_shap_values=True)
    # print('gshap_preds', gshap_preds)
    print('MSE global error train', ((gshap_preds - outputs_train) ** 2).mean())
    # print('gshap_vals', gshap_vals)

    gshap_preds, gshap_vals = gshap.predict(data_test, return_shap_values=True)
    outputs_test = model(data_test)
    print('MSE global error test', ((gshap_preds - outputs_test) ** 2).mean())


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
