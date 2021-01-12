"""
explain_classifier_test.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import random
from itertools import combinations
from itertools import chain

import pandas as pd
import numpy as np
from scipy.special import comb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
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

scaler = StandardScaler()
X = scaler.fit_transform(X)

random.seed(5318008)

max_order = 2
start_interact_order = 0
# start_interact_order = 3
# n_main = 4
n_main = X.shape[1]
n_interact_max = 0

model_type = 'dnn'
n_units = 16
activation = 'relu'

if model_type == 'dnn':
    fit_kwargs = {'epochs': 1}

else:
    fit_kwargs = {}

terms = []
features = []
# terms = [T.te(0, 1), T.te(2, 3), T.s(0, n_splines=50)]
# terms = [T.te(0, 1), T.te(1, 3, n_splines=5), T.s(2, n_splines=50)]

if not terms:
    for order in range(1, max_order + 1):
        if order == 1:
            for i in random.sample(range(X.shape[1]), k=n_main):
                if model_type == 'dnn':
                    terms.append([
                        Lambda(lambda x: x[:, i:(i + 1)], output_shape=[1]),
                        Dense(n_units, activation=activation),
                        Dense(1, activation=activation),
                    ])
                elif model_type == 'gam':
                    terms.append(T.s(i, n_splines=25))
                features.append((i,))
        elif order >= start_interact_order:
            n_interact = min(n_interact_max - len(terms) + n_main,
                             comb(X.shape[1], order))
            selected_interact = random.sample(
                [*combinations(range(X.shape[1]), order)], k=n_interact)
            for feats in selected_interact:
                if model_type == 'dnn':
                    terms.append([
                        Lambda(lambda x: tf.gather(x, feats, axis=1),
                               output_shape=[len(feats)]),
                        # TODO: 2 * units? meh
                        Dense(n_units, activation=activation),
                        Dense(1, activation=activation),
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

# assert all((len(terms) + 1) == len(e.terms)
#            for e in model._estimators)

explain_only_this_many = 50
X_trunc = X[:explain_only_this_many]

contribs = model.feature_contributions(X_trunc)

explainer = KernelSHAPExplainer(model, task=task,
                                n_cpus=1 if model_type == 'dnn' else -1)
explainer.fit(X)  # fit full X
explanation = explainer.feature_contributions(X_trunc, as_dict=True)

assert len(explanation) == len(contribs)


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


rows = []
for i, (e_true_i, e_pred_i) in enumerate(zip(contribs, explanation)):
    # shed zero-elements
    e_pred_i = standardize_contributions(e_pred_i)
    components, goodness = generous_eval(e_true_i, e_pred_i)
    matches = apply_matching(components, e_true_i, e_pred_i, len(X_trunc))

    for ((true_feats, pred_feats),
         (true_contrib_i, pred_contrib_i)) in matches.items():

        all_feats = [*{*chain(chain.from_iterable(true_feats),
                              chain.from_iterable(pred_feats))}]

        if len(all_feats) > 1:
            print(f'skipping match with {all_feats} for now as is interaction '
                  f'true_feats {true_feats} | pred_feats {pred_feats}')
            continue
        xi = X_trunc[:, all_feats[0]]  # TODO(interactions)
        base = {
            'class': i,
            'true_effect': true_feats,
            'pred_effect': pred_feats,
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

sns.relplot(
    data=df,
    x='feature value',
    y='contribution',
    hue='explainer',
    col='class',
    row='true_effect',
    kind='scatter',
    # x_jitter=.05,
)

if plt.get_backend() == 'agg':
    plt.savefig(f'contributions_grid_{model_type}.pdf')
else:
    plt.show()
