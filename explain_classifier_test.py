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

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from posthoceval.explainers.local.shap import KernelSHAPExplainer
from posthoceval.explainers.local.maple import MAPLEExplainer
from posthoceval.explainers.local.lime import LIMEExplainer
from posthoceval.models.gam import MultiClassLogisticGAM
from posthoceval.models.gam import T
from posthoceval.models.dnn import DNNRegressor
from posthoceval.metrics import generous_eval
from posthoceval.metrics import standardize_contributions
from posthoceval.utils import atomic_write_exclusive
from posthoceval.utils import nonexistent_filename
from posthoceval import metrics

sns.set_theme(
    context='paper',
    # context='notebook',
    style='ticks',
    font_scale=1.5,
    color_codes=True,
    # palette=sns.color_palette('pastel'),
)

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
    task = 'regression'
    data_df = pd.read_csv('data/boston', delimiter=' ')
    label_col = 'MEDV'

    X_df = data_df.drop(columns=label_col)
    X = X_df.values
    y = data_df[label_col].values

    headers = [*X_df.keys()]
else:
    task = 'classification'
    # dataset = datasets.load_iris()
    # dataset = datasets.load_breast_cancer()
    dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target

X = np.asarray(X)
y = np.asarray(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_scaler = None
if task == 'regression':
    y_scaler = StandardScaler()
    if y.ndim < 2:
        y = y[:, np.newaxis]
    y = y_scaler.fit_transform(y)
    y = y.squeeze(axis=1)

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

    # TODO..
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

    if task == 'classification':
        terms = sum(terms[1:], terms[0])
        model = MultiClassLogisticGAM(
            symbols=symbols, terms=terms, max_iter=100, verbose=True
        )
    else:
        raise NotImplementedError(task)

model.fit(X, y, **fit_kwargs)

if model_type == 'dnn':
    model.plot_model(nonexistent_filename('dnn.png'),
                     show_shapes=True)

# explain_only_this_many = 512
explain_only_this_many = 101
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

# if 1:
#     explainer_name = 'SHAP'
#     explainer = KernelSHAPExplainer(model, task=task, seed=seed,
#                                     n_cpus=1 if model_type == 'dnn' else -1)
# elif 0:
#     explainer_name = 'LIME'
#     explainer = LIMEExplainer(model, seed=seed, task=task)
# else:
#     explainer_name = 'MAPLE'
#     explainer = MAPLEExplainer(model, seed=seed, task=task)

rows = []
rows_3d = []

for expl_i, (explainer_name, explainer) in enumerate((
        ('SHAP',
         KernelSHAPExplainer(model, task=task, seed=seed,
                             n_cpus=1 if model_type == 'dnn' else -1)),
        ('LIME',
         LIMEExplainer(model, seed=seed, task=task)),
        ('MAPLE',
         MAPLEExplainer(model, seed=seed, task=task)),
)):
    print('Start explainer', explainer_name)

    explainer.fit(X)  # fit full X
    intercepts = None
    y_expl = None
    if explainer_name == 'LIME' or explainer_name == 'MAPLE':
        explanation, intercepts = explainer.feature_contributions(
            X_trunc, as_dict=True, return_intercepts=True)
    elif explainer_name == 'MAPLE':
        explanation, y_expl = explainer.feature_contributions(
            X_trunc, as_dict=True, return_y=True)
    else:
        explanation = explainer.feature_contributions(X_trunc, as_dict=True)

    nrmse_func = metrics.nrmse_interquartile
    # nrmse_func = metrics.nrmse_range

    if task == 'regression':
        y_pred = model(X)
        contribs_full = model.feature_contributions(X)

        print('GT vs. NN')
        print(f' RMSE={metrics.rmse(y, y_pred)}')
        print(f'NRMSE={nrmse_func(y, y_pred)}')

        print('NN Out vs. NN Contribs')
        y_contrib_pred = np.asarray([*contribs_full.values()]).sum(axis=0)
        print(f' RMSE={metrics.rmse(y_pred, y_contrib_pred)}')
        print(f'NRMSE={nrmse_func(y_pred, y_contrib_pred)}')

        print('NN vs. Explainer')
        y_pred_trunc = model(X_trunc)
        if y_expl is None:
            y_expl = np.asarray([*explanation.values()]).sum(axis=0)
            if intercepts is not None:
                y_expl += np.asarray(intercepts)
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
                   label='NN')
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


    def apply_matching(matching, true_expl, pred_expl, n_explained):
        matches = {}
        for match_true, match_pred in matching:
            if match_true:
                contribs_true = sum(
                    [true_expl[effect] for effect in match_true])
                contribs_true_mean = np.mean(contribs_true)
            else:
                contribs_true = contribs_true_mean = np.zeros(n_explained)
            if match_pred:
                # add the mean back for these effects (this will be the
                #  same sample mean that the explainer saw before)
                contribs_pred = sum(
                    [pred_expl[effect] for effect in match_pred],
                    start=contribs_true_mean
                )
            else:
                contribs_pred = np.zeros(n_explained)

            match_key = (tuple(match_true), tuple(match_pred))
            matches[match_key] = (contribs_true, contribs_pred)

        return matches


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
        matches = apply_matching(components, e_true_i, e_pred_i, len(X_trunc))
        print(matches)

        true_func_idx = pred_func_idx = 1
        for ((true_feats, pred_feats),
             (true_contrib_i, pred_contrib_i)) in matches.items():

            all_feats = [*{*chain(chain.from_iterable(true_feats),
                                  chain.from_iterable(pred_feats))}]

            # TODO non-logit...
            contribution = pred_contrib_i + true_contrib_i.mean()

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
            xi = X_trunc[:, f_idxs]
            base = {
                'class': i,
                'true_effect': true_feats,
                'pred_effect': pred_feats,
                'Match': match_str,
            }
            true_row = base.copy()  # TODO: true last...
            true_row['explainer'] = 'True'

            pred_row = base
            pred_row['contribution'] = contribution
            pred_row['explainer'] = explainer_name

            for true_contrib_ik, pred_contrib_ik, xik in zip(
                    true_contrib_i, pred_contrib_i, xi):
                true_row_i = true_row.copy()
                true_row_i['contribution'] = true_contrib_ik

                pred_row_i = pred_row.copy()
                pred_row_i['contribution'] = pred_contrib_ik

                if len(all_feats) == 1:
                    pred_row_i['feature value'] = xik[0]
                    rows.append(pred_row_i)

                    # TODO: wow...(3 explainers total...)
                    if (expl_i + 1) == 3:
                        true_row_i['feature value'] = xik[0]
                        rows.append(true_row_i)
                else:  # interaction == 2
                    pred_row_i['feature value x'] = xik[0]
                    pred_row_i['feature value y'] = xik[1]
                    rows_3d.append(pred_row_i)

                    # TODO: wow...(3 explainers total...)
                    if (expl_i + 1) == 3:
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
