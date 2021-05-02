"""
explain_classifier_test.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
# maximize reproducibility: set seed with minimal imports
# just a seed
seed = 431136
import os

# verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# reproducibility
# https://github.com/NVIDIA/framework-determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random

random.seed(seed)
rng_r = random.Random(seed)

import numpy as np

np.random.seed(seed)
rng_np = np.random.default_rng(seed)

import tensorflow as tf

tf.random.set_seed(seed)

from itertools import combinations
from itertools import chain

import pandas as pd
from scipy.special import comb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from posthoceval.explainers.local.shap import KernelSHAPExplainer
from posthoceval.expl_utils import apply_matching, standardize_contributions
from posthoceval.metrics import generous_eval
from posthoceval import metrics
from posthoceval.models.gam import MultiClassLogisticGAM
from posthoceval.models.gam import LinearGAM
from posthoceval.models.gam import T
from posthoceval.models.dnn import DNNRegressor
from posthoceval.transform import Transformer
from posthoceval.utils import nonexistent_filename
from posthoceval.datasets.boston import BostonDataset
from posthoceval.datasets.compas import COMPASDataset

sns.set_theme(
    context='paper',
    # context='notebook',
    style='ticks',
    font_scale=2.25,
    color_codes=True,
    # palette=sns.color_palette('pastel'),
)

scaler = None
categorical_cols = None

if 0:
    task = 'regression'

    # import numpy
    # X = np.random.rand(1000, 8) / 4
    # x1, x2, x3, x4, x5, x6, x7, x8 = X.T
    # y = x1**2 + x5**2 + x5*numpy.log(x1 + x2) + x7*numpy.select([numpy.greater(x2, numpy.sinc(x1/numpy.pi)),True], [numpy.asarray(x2**(-1.0)).astype(numpy.bool),numpy.asarray(numpy.sinc(x1/numpy.pi)**(-1.0)).astype(numpy.bool)], default=numpy.nan) + (x1*abs(x7) + x5)**3 + numpy.exp(x7) + numpy.exp((x1 + x2)/x5) + numpy.sin(numpy.log(x2))

    # X = np.random.randn(1000, 4)
    # x1, x2, x3, x4 = X.T
    # x1 = np.abs(x1)
    # x2 = np.abs(x2)
    # y = x1 ** (1 / 4) + np.sqrt(x2) + np.exp(x3 / 2) + np.abs(x4) + np.tan(x4) / x1 ** 2

    # X = np.random.randn(1000, 2)
    # y = X[:, 0] ** 9 + np.tan(X[:, 1]) + np.abs(X[:, 0] / X[:, 1] ** 2)

    # X = np.random.randn(1000, 400)
    # y = np.exp(np.random.randn(len(X)))

    # X[:, 1] = X[:, 0] / 2
    # X[:, 2] = X[:, 1] + 1
    # X[:, 3] = X[:, 2] * 2.6
    # y = (np.sin(X[:, 0] ** 3) + np.maximum(X[:, 1], 0)
    #     - np.sin(X[:, 2]) / X[:, 2] + 2 * X[:, 3])

    headers = [*range(X.shape[1])]
elif 1:
    dataset_cls = COMPASDataset
elif 1:
    dataset_cls = BostonDataset
else:
    task = 'classification'
    # dataset = datasets.load_iris()
    # dataset = datasets.load_breast_cancer()
    dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target

# load dataset
dataset_orig = dataset_cls()
# transform data
transformer = Transformer()
dataset = transformer.fit_transform(dataset_orig)
# extract data
task = dataset.task
X = dataset.X
y = dataset.y
headers = dataset.feature_names

desired_interactions = []

# current interact plots use this: LIME, MAPLE
# desired_interactions = [(1, 2)]

# features 8 & 9 correlate in Boston dataset
# desired_interactions = [(8, 0, 1), (2, 8), (2, 9)]
# desired_interactions = [(5, 8), (5, 9)]
# desired_interactions = [(2, 8), (5, 9)]

# desired_interactions = [(1, 2), (4, 9), (8, 10)]

# desired_interactions = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]
# desired_interactions = [(0, 1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]
# desired_interactions = [(0, 1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12),
#                         (220, 101)]

max_order = 2
start_interact_order = 0
# start_interact_order = 3
# n_main = 4
n_main = X.shape[1]
n_interact_max = 0 or len(desired_interactions)

# model_type = 'gam'
model_type = 'dnn'

n_units = 64
activation = 'relu'

if model_type == 'dnn':
    callback = EarlyStopping(monitor='loss', mode='min', patience=100,
                             restore_best_weights=True)
    optimizer = Adam(learning_rate=1e-3)
    fit_kwargs = {'epochs': 1000, 'batch_size': len(X),
                  'callbacks': [callback], 'optimizer': optimizer}
else:
    fit_kwargs = {}

terms = []
features = []
# terms = [T.te(0, 1), T.te(2, 3), T.s(0, n_splines=50)]
# terms = [T.te(0, 1), T.te(1, 3, n_splines=5), T.s(2, n_splines=50)]

# TODO: factor terms for categoricals in GAM?
# TODO: embed categoricals in NN?
if not terms:
    for order in range(1, max_order + 1):
        if order == 1:
            for i in rng_r.sample(range(X.shape[1]), k=n_main):

                if model_type == 'dnn':

                    base_name = f'branch_main/feature_{i}_'
                    terms.append([
                        # Lambda(lambda x: x[:, i:(i + 1)], output_shape=[1],
                        #        name=base_name + 'split'),
                        Dense(n_units, activation=activation,
                              name=base_name + f'l1_d{n_units}'),
                        Dense(n_units, activation=activation,
                              name=base_name + f'l2_d{n_units}'),
                        Dense(n_units // 2, activation=activation,
                              name=base_name + f'l3_d{n_units // 2}'),
                        Dense(1, activation=None,
                              name=base_name + 'linear_d1'),
                    ])

                elif model_type == 'gam':

                    terms.append(T.s(i, n_splines=25))

                features.append((i,))

        elif order >= start_interact_order:

            n_interact = min(n_interact_max - len(terms) + n_main,
                             comb(X.shape[1], order))
            if desired_interactions is None:
                selected_interact = rng_r.sample(
                    [*combinations(range(X.shape[1]), order)], k=n_interact)
            else:
                selected_interact = desired_interactions

            for feats in selected_interact:

                if model_type == 'dnn':

                    feats_str = '_'.join(map(str, feats))
                    base_name = f'branch_interact/features_{feats_str}_'
                    terms.append([
                        # Lambda(lambda x: tf.gather(x, feats, axis=1),
                        #        output_shape=[len(feats)],
                        #        name=base_name + 'split'),
                        # TODO: 2 * units? meh
                        Dense(n_units, activation=activation,
                              name=base_name + f'l1_d{n_units}'),
                        Dense(n_units, activation=activation,
                              name=base_name + f'l2_d{n_units}'),
                        Dense(n_units // 2, activation=activation,
                              name=base_name + f'l3_d{n_units // 2}'),
                        Dense(1, activation=None,
                              name=base_name + 'linear_d1'),
                    ])

                elif model_type == 'gam':

                    terms.append(T.te(*feats, n_splines=10))

                features.append(tuple(feats))

            if desired_interactions is not None:
                break

symbols = [*range(1, X.shape[1] + 1)]
if model_type == 'dnn':

    # TODO...
    assert task == 'regression'

    x = Input([X.shape[1]])

    outputs = []
    output_map = {}
    for branch, feats in zip(terms, features):
        xl = tf.gather(x, feats, axis=1)
        # xl = K.print_tensor(xl, message=f'branch {bi} input=')
        for layer in branch:
            xl = layer(xl)
        outputs.append(xl)
        feat_symbols = tuple(symbols[fi] for fi in feats)
        output_map[feat_symbols] = xl

    output = Add()(outputs)
    tf_model = Model(x, output)
    model = DNNRegressor(
        tf_model, output_map, symbols=symbols,
    )

elif model_type == 'gam':

    terms = sum(terms[1:], terms[0])

    if task == 'classification':
        model = MultiClassLogisticGAM(
            symbols=symbols, terms=terms, max_iter=100, verbose=True
        )
    else:
        model = LinearGAM(
            symbols=symbols, terms=terms, max_iter=100, verbose=True
        )

model.fit(X, y, **fit_kwargs)

if model_type == 'dnn':
    model.plot_model(nonexistent_filename('dnn.png'),
                     show_shapes=True)

explain_only_this_many = 512
# explain_only_this_many = 101
# explain_only_this_many = 12
# explain_only_this_many = len(X)
explain_only_this_many = min(explain_only_this_many, len(X))
sample_idxs_all = np.arange(len(X))
sample_idxs = rng_np.choice(sample_idxs_all,
                            size=explain_only_this_many, replace=False)
X_trunc = X[sample_idxs]

contribs = model.feature_contributions(X_trunc)

if task == 'regression':
    contribs = [contribs]

rows = []
rows_3d = []

explainer_array = (
    # ('LIME',
    #  LIMEExplainer(model, seed=seed, task=task)),
    # ('MAPLE',
    #  MAPLEExplainer(model, seed=seed, task=task)),
    ('SHAP',
     KernelSHAPExplainer(model, task=task, seed=seed,
                         n_cpus=1 if model_type == 'dnn' else -1,
                         **expl_init_kwargs)),
)

# TODO: feature_contributions() --> explain()
# TODO: explain() --> ExplainerMixin (for both models and explainers)

for expl_i, (explainer_name, explainer) in enumerate(explainer_array):
    print('Start explainer', explainer_name)
    explainer.fit(X)  # fit full X
    explanation, y_expl = explainer.feature_contributions(
        X_trunc, as_dict=True, return_predictions=True)

    nrmse_func = metrics.nrmse_interquartile
    # nrmse_func = metrics.nrmse_range

    if task == 'regression':
        y_pred = model(X)

        # TODO: unify
        model_intercepts = 0
        if model_type == 'gam':
            contribs_full, model_intercepts = model.feature_contributions(
                X, return_intercepts=True)
        else:
            contribs_full = model.feature_contributions(X)

        print(f'GT vs. {model_type}')
        print(f' RMSE={metrics.rmse(y, y_pred)}')
        print(f'NRMSE={nrmse_func(y, y_pred)}')

        # This should be 0
        print(f'{model_type} Out vs. {model_type} Contribs')
        y_contrib_pred = np.asarray([*contribs_full.values()]).sum(axis=0)
        y_contrib_pred += model_intercepts
        print(f' RMSE={metrics.rmse(y_pred, y_contrib_pred)}')
        print(f'NRMSE={nrmse_func(y_pred, y_contrib_pred)}')

        print(f'{model_type} vs. Explainer')
        y_pred_trunc = model(X_trunc)
        print(f' RMSE={metrics.rmse(y_pred_trunc, y_expl)}')
        print(f'NRMSE={nrmse_func(y_pred_trunc, y_expl)}')

        fig, ax = plt.subplots()
        ax.scatter(sample_idxs_all,
                   y,
                   alpha=.65,
                   label='GT')
        ax.scatter(sample_idxs_all,
                   # sample_idxs,
                   y_pred,
                   # y_pred_trunc,
                   alpha=.65,
                   label=f'{model_type}')
        ax.scatter(sample_idxs,
                   y_expl,
                   alpha=.65,
                   label='Explainer')
        ax.set_xlabel('Sample idx')
        ax.set_ylabel('Predicted value')
        fig.legend()

        if plt.get_backend() == 'agg':
            fig.savefig(
                nonexistent_filename(
                    f'prediction_comparison_{model_type}_{explainer_name}.pdf'
                )
            )
        else:
            plt.show()


    def make_tex_str(features, start_i, explained=False):
        out_strs = []
        for feats in features:
            feats_str = ','.join(
                f'x_{{{feat}}}' if isinstance(feat, int) else str(feat)
                for feat in feats
            )
            if explained:
                out_str = fr'\hat{{f}}_{{{start_i}}}({feats_str})'
            else:
                out_str = f'f_{{{start_i}}}({feats_str})'
            out_strs.append(out_str)
            start_i += 1
        return '$' + ('+'.join(out_strs) or '0') + '$'


    if task == 'regression':
        explanation = [explanation]
    else:
        assert len(explanation) == len(contribs)

    for i, (e_true_i, e_pred_i) in enumerate(zip(contribs, explanation)):
        # shed zero-elements
        e_true_i = standardize_contributions(e_true_i)
        e_pred_i = standardize_contributions(e_pred_i)
        components, goodness = generous_eval(e_true_i, e_pred_i)
        matches = apply_matching(components, e_true_i, e_pred_i, len(X_trunc),
                                 explainer_name)
        # print(matches)

        true_func_idx = pred_func_idx = 1
        for ((true_feats, pred_feats),
             (true_contrib_i, pred_contrib_i)) in matches.items():

            # TODO: blegh
            true_contrib_is_zero = (true_contrib_i == 0.).all()
            pred_contrib_is_zero = (pred_contrib_i == 0.).all()

            all_feats = [*{*chain(chain.from_iterable(true_feats),
                                  chain.from_iterable(pred_feats))}]
            f_idxs = [model.symbols.index(fi) for fi in all_feats]

            feature_str = ' & '.join(map(str, (headers[fi] for fi in f_idxs)))

            match_str = (
                feature_str  # + '\n' +
                # TODO: depression
                # 'True: ' +
                # make_tex_str(true_feats, true_func_idx, False) +
                # ' | Predicted: ' +
                # ' vs. ' +
                # make_tex_str(pred_feats, pred_func_idx, True)
            )
            true_func_idx += len(true_feats)
            pred_func_idx += len(pred_feats)

            print(match_str, ' RMSE',
                  metrics.rmse(true_contrib_i, pred_contrib_i))
            nrmse_score = nrmse_func(true_contrib_i, pred_contrib_i)
            print(match_str, 'NRMSE', nrmse_score)
            print()

            # pretty format score
            # TODO: sad times we have here
            # match_str += ('\nNRMSE = ' + (f'{nrmse_score:.3f}'
            #                               if (1e-3 < nrmse_score < 1e3) else
            #                               f'{nrmse_score:.3}'))

            if len(all_feats) > 2:
                print(
                    f'skipping match with {all_feats} for now as is interaction '
                    f'with order > 2 true_feats {true_feats} | pred_feats '
                    f'{pred_feats}')
                continue
            X_trunc_inverse = X_trunc
            # TODO!!!inverse_transform
            if scaler is not None and hasattr(scaler, 'inverse_transform'):
                X_trunc_inverse = scaler.inverse_transform(X_trunc_inverse)
            xi = X_trunc_inverse[:, f_idxs]
            base = {
                'class': i,
                'true_effect': true_feats,
                'pred_effect': pred_feats,
                'Match': match_str,
            }
            true_row = base.copy()
            true_row['explainer'] = 'True'

            pred_row = base
            pred_row['explainer'] = explainer_name

            for true_contrib_ik, pred_contrib_ik, xik in zip(
                    true_contrib_i, pred_contrib_i, xi):
                true_row_i = true_row.copy()
                if scale_y is not None and y.ndim == 1:
                    true_contrib_ik = float(scale_y(
                        y_scaler.inverse_transform, true_contrib_ik))
                true_row_i['contribution'] = true_contrib_ik

                pred_row_i = pred_row.copy()
                if scale_y is not None and y.ndim == 1:
                    pred_contrib_ik = float(scale_y(
                        y_scaler.inverse_transform, pred_contrib_ik))
                pred_row_i['contribution'] = pred_contrib_ik

                if len(all_feats) == 1:
                    if not pred_contrib_is_zero:
                        pred_row_i['feature value'] = xik[0]
                        rows.append(pred_row_i)

                    if (not true_contrib_is_zero and
                            (expl_i + 1) == len(explainer_array)):
                        true_row_i['feature value'] = xik[0]
                        rows.append(true_row_i)
                else:  # interaction == 2
                    if not pred_contrib_is_zero:
                        pred_row_i['feature value x'] = xik[0]
                        pred_row_i['feature value y'] = xik[1]
                        rows_3d.append(pred_row_i)

                    if (not true_contrib_is_zero and
                            (expl_i + 1) == len(explainer_array)):
                        true_row_i['feature value x'] = xik[0]
                        true_row_i['feature value y'] = xik[1]
                        rows_3d.append(true_row_i)

df = pd.DataFrame(rows)

col_wrap = 4

if not df.empty:
    g = sns.relplot(
        data=df,
        x='feature value',
        y='contribution',
        hue='explainer',
        # col='class' if task == 'classification' else 'true_effect',
        col='class' if task == 'classification' else 'Match',
        col_wrap=None if task == 'classification' else col_wrap,
        # row='true_effect' if task == 'classification' else None,
        row='Match' if task == 'classification' else None,
        kind='scatter',
        x_jitter=.08,  # for visualization purposes of nearby points
        alpha=.65,
        facet_kws=dict(sharex=False, sharey=False),
    )
    for ax in g.axes.flat:
        title = ax.get_title()
        ax.set_title(title.split(' = ', 1)[1])

# 3d interaction plot time TODO this is regression-only atm...
df_3d = pd.DataFrame(rows_3d)

if not df_3d.empty:

    plt_x = 'feature value x'
    plt_y = 'feature value y'
    plt_z = 'contribution'
    plt_hue = 'explainer'
    plt_col = 'Match'

    df_3d_grouped = df_3d.groupby(['class', plt_col])

    n_plots = len(df_3d_grouped)
    n_rows = int(np.ceil(n_plots / col_wrap))
    n_cols = min(col_wrap, n_plots)
    figsize = plt.rcParams['figure.figsize']
    figsize = (figsize[0] * n_cols, figsize[1] * n_rows)
    fig = plt.figure(figsize=figsize)

    for i, ((class_i, ax_title), group_3d) in enumerate(df_3d_grouped):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

        for hue_name, hue_df in group_3d.groupby(plt_hue):
            ax.scatter(
                hue_df[plt_x],
                hue_df[plt_y],
                hue_df[plt_z],
                label=hue_name,
                alpha=.5,
            )
        ax.set_xlabel(plt_x)
        ax.set_ylabel(plt_y)
        ax.set_zlabel(plt_z)

        ax.set_title(ax_title)

        if i == 0:
            fig.legend(loc='center right')

if not df.empty:
    g.savefig(nonexistent_filename(f'contributions_grid_{model_type}.pdf'))
if not df_3d.empty:
    fig.savefig(nonexistent_filename(
        f'contributions_grid_interact_{model_type}.pdf'))

if plt.get_backend() != 'agg':
    plt.show()
