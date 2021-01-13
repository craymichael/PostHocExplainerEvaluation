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
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from posthoceval.explainers.local.shap import KernelSHAPExplainer
from posthoceval.models.gam import MultiClassLogisticGAM
from posthoceval.models.gam import T
from posthoceval.models.dnn import DNNRegressor
from posthoceval.metrics import generous_eval
from posthoceval.metrics import standardize_contributions
from posthoceval.utils import atomic_write_exclusive
from posthoceval.utils import nonexistent_filename
from posthoceval import metrics

sns.set()

if 1:
    task = 'regression'
    data_df = pd.read_csv('data/boston', delimiter=' ')
    label_col = 'MEDV'

    X = data_df.drop(columns=label_col).values
    y = data_df[label_col].values
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

max_order = 2
start_interact_order = 0
# start_interact_order = 3
# n_main = 4
n_main = X.shape[1]
n_interact_max = 0

model_type = 'dnn'
n_units = 32
activation = 'relu'

if model_type == 'dnn':
    fit_kwargs = {'epochs': 10}

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
                        Lambda(lambda x: x[:, i:(i + 1)], output_shape=[1],
                               name=base_name + 'split'),
                        Dense(n_units, activation=activation,
                              name=base_name + f'd{n_units}'),
                        Dense(1, activation=activation,
                              name=base_name + 'd1'),
                    ])
                elif model_type == 'gam':
                    terms.append(T.s(i, n_splines=25))
                features.append((i,))
        elif order >= start_interact_order:
            n_interact = min(n_interact_max - len(terms) + n_main,
                             comb(X.shape[1], order))
            selected_interact = rng_r.sample(
                [*combinations(range(X.shape[1]), order)], k=n_interact)
            for feats in selected_interact:
                if model_type == 'dnn':
                    feats_str = '_'.join(map(str, feats))
                    base_name = f'branch_interact/features_{feats_str}_'
                    terms.append([
                        Lambda(lambda x: tf.gather(x, feats, axis=1),
                               output_shape=[len(feats)],
                               name=base_name + 'split'),
                        # TODO: 2 * units? meh
                        Dense(n_units, activation=activation,
                              name=base_name + f'd{n_units}'),
                        Dense(1, activation=activation,
                              name=base_name + 'd1'),
                    ])
                elif model_type == 'gam':
                    terms.append(T.te(*feats, n_splines=10))

                features.append(tuple(feats))

if model_type == 'dnn':

    x = Input([X.shape[1]])

    outputs = []
    output_map = {}
    for branch, feats in zip(terms, features):
        xl = x
        for layer in branch:
            xl = layer(xl)
        outputs.append(xl)
        output_map[feats] = xl

    output = Add()(outputs)
    tf_model = Model(x, output)
    model = DNNRegressor(
        tf_model, output_map, symbols=[*range(X.shape[1])],
    )

elif model_type == 'gam':

    terms = sum(terms[1:], terms[0])
    model = MultiClassLogisticGAM(
        symbols=[*range(X.shape[1])], terms=terms, max_iter=100, verbose=True
    )

model.fit(X, y, **fit_kwargs)

if model_type == 'dnn':
    model.plot_model(nonexistent_filename('dnn.png'))

explain_only_this_many = 101
# explain_only_this_many = len(X)
X_trunc = rng_np.choice(X, size=explain_only_this_many, replace=False)

contribs = model.feature_contributions(X_trunc)

explainer = KernelSHAPExplainer(model, task=task,
                                n_cpus=1 if model_type == 'dnn' else -1)
explainer.fit(X)  # fit full X
explanation = explainer.feature_contributions(X_trunc, as_dict=True)

if task == 'regression':
    y_pred = model(X)
    contribs_full = model.feature_contributions(X)

    print('GT vs. NN')
    print(f' RMSE={metrics.rmse(y, y_pred)}')
    print(f'NRMSE={metrics.nrmse_interquartile(y, y_pred)}')

    print('NN Out vs. NN Contribs')
    y_contrib_pred = np.asarray([*contribs_full.values()]).sum(axis=0)
    print(f' RMSE={metrics.rmse(y_pred, y_contrib_pred)}')
    print(f'NRMSE={metrics.nrmse_interquartile(y_pred, y_contrib_pred)}')

    print('NN vs. Explainer')
    y_pred_trunc = model(X_trunc)
    y_expl = np.asarray([*explanation.values()]).sum(axis=0)
    print(f' RMSE={metrics.rmse(y_pred_trunc, y_expl)}')
    print(f'NRMSE={metrics.nrmse_interquartile(y_pred_trunc, y_expl)}')

    fig, ax = plt.subplots()
    sample_idxs = np.arange(explain_only_this_many)
    sample_idxs_all = np.arange(len(X))
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
            nonexistent_filename(f'contributions_grid_{model_type}.pdf'))
    else:
        plt.show()


# TODO: make this func else where?
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


rows = []

if task == 'regression':
    contribs = [contribs]
    explanation = [explanation]
else:
    assert len(explanation) == len(contribs)

for i, (e_true_i, e_pred_i) in enumerate(zip(contribs, explanation)):
    # shed zero-elements
    e_true_i = standardize_contributions(e_true_i)
    e_pred_i = standardize_contributions(e_pred_i)
    components, goodness = generous_eval(e_true_i, e_pred_i)
    matches = apply_matching(components, e_true_i, e_pred_i, len(X_trunc))

    true_func_idx = pred_func_idx = 1
    for ((true_feats, pred_feats),
         (true_contrib_i, pred_contrib_i)) in matches.items():

        all_feats = [*{*chain(chain.from_iterable(true_feats),
                              chain.from_iterable(pred_feats))}]

        if len(all_feats) > 1:
            print(f'skipping match with {all_feats} for now as is interaction '
                  f'true_feats {true_feats} | pred_feats {pred_feats}')
            continue
        xi = X_trunc[:, all_feats[0]]  # TODO(interactions)
        match_str = (
                'True: ' +
                make_tex_str(true_feats, true_func_idx, False) +
                ' | Predicted: ' +
                make_tex_str(pred_feats, pred_func_idx, True)
        )
        true_func_idx += len(true_feats)
        pred_func_idx += len(pred_feats)
        base = {
            'class': i,
            'true_effect': true_feats,
            'pred_effect': pred_feats,
            'Match': match_str,
        }
        true_row = base.copy()
        true_row['explainer'] = 'True'

        pred_row = base
        # TODO non-logit...
        pred_row['contribution'] = pred_contrib_i + true_contrib_i.mean()
        pred_row['explainer'] = 'SHAP'

        for true_contrib_ik, pred_contrib_ik, xik in zip(
                true_contrib_i, pred_contrib_i, xi):
            true_row_i = true_row.copy()
            true_row_i['contribution'] = true_contrib_ik
            true_row_i['feature value'] = xik
            rows.append(true_row_i)

            pred_row_i = pred_row.copy()
            pred_row_i['contribution'] = pred_contrib_ik
            pred_row_i['feature value'] = xik
            rows.append(pred_row_i)

df = pd.DataFrame(rows)

g = sns.relplot(
    data=df,
    x='feature value',
    y='contribution',
    hue='explainer',
    # col='class' if task == 'classification' else 'true_effect',
    col='class' if task == 'classification' else 'Match',
    col_wrap=None if task == 'classification' else 4,
    # row='true_effect' if task == 'classification' else None,
    row='Match' if task == 'classification' else None,
    kind='scatter',
    # x_jitter=.05,  # for visualization purposes of nearby points
    alpha=.65,
    facet_kws=dict(sharex=False, sharey=False),
)

if plt.get_backend() == 'agg':
    g.savefig(nonexistent_filename(f'contributions_grid_{model_type}.pdf'))
else:
    plt.show()