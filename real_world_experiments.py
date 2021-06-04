


import warnings

seed = 431136
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random

random.seed(seed)
rng_r = random.Random(seed + 1)

import numpy as np

np.random.seed(seed + 2)
rng_np = np.random.default_rng(seed + 3)

import tensorflow as tf

tf.random.set_seed(seed + 4)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import seaborn as sns

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from posthoceval.explainers import KernelSHAPExplainer
from posthoceval.models.gam import MultiClassLogisticGAM
from posthoceval.models.gam import LinearGAM
from posthoceval.models.gam import T
from posthoceval.models.dnn import AdditiveDNN
from posthoceval.transform import Transformer
from posthoceval.utils import nonexistent_filename
from posthoceval.datasets import COMPASDataset
from posthoceval.datasets import BostonDataset
from posthoceval.models.term_util import generate_terms
from posthoceval.viz import gather_viz_data

sns.set_theme(
    context='paper',
    
    style='ticks',
    font_scale=2.25,
    color_codes=True,
    
)


dataset_name = 'compas'

if dataset_name == 'synthetic':
    raise NotImplementedError
    task = 'regression'

    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    

    
    

    
    

    
    
    
    
    

    feature_names = [*range(X.shape[1])]
elif dataset_name == 'compas':
    dataset_cls = COMPASDataset
elif dataset_name == 'boston':
    dataset_cls = BostonDataset
else:
    raise NotImplementedError
    task = 'classification'
    
    
    dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target


dataset_orig = dataset_cls()


transformer = Transformer()
dataset = transformer.fit_transform(dataset_orig)


task = dataset.task
X = dataset.X
y = dataset.y
feature_names = dataset.feature_names
n_features = dataset.n_features



model_type = 'dnn'


desired_interactions = []
















max_order = 2
min_order = 1
n_main = n_features
n_interact = len(desired_interactions) if desired_interactions else None

terms = generate_terms(
    n_features=n_features,
    n_main=n_main,
    n_interact=n_interact,
    desired_interactions=desired_interactions,
    min_order=min_order,
    max_order=max_order,
    seed=rng_np,
)



if model_type == 'dnn':
    callback = EarlyStopping(monitor='loss', mode='min', patience=100,
                             restore_best_weights=True)
    optimizer = Adam(learning_rate=1e-3)
    fit_kwargs = {'epochs': 1000, 'batch_size': len(X),
                  'callbacks': [callback], 'optimizer': optimizer}
else:
    fit_kwargs = {}




if model_type == 'dnn':
    model = AdditiveDNN(
        terms=terms,
        task=task,
        symbol_names=feature_names,
    )
elif model_type == 'gam':
    if task == 'classification':
        model = MultiClassLogisticGAM(symbol_names=feature_names, terms=terms)
    else:
        model = LinearGAM(symbol_names=feature_names, terms=terms)
else:
    raise NotImplementedError(model_type)

model.fit(X, y, **fit_kwargs)

if model_type == 'dnn':
    model.plot_model(nonexistent_filename('dnn.png'),
                     show_shapes=True)

explain_only_this_many = 512



explain_only_this_many = min(explain_only_this_many, len(X))
sample_idxs_all = np.arange(len(X))
sample_idxs = rng_np.choice(sample_idxs_all,
                            size=explain_only_this_many, replace=False)
X_subset = X[sample_idxs]

true_contribs = model.feature_contributions(X_subset)


explainer_array = (
    
    
    
    
    ('SHAP',
     KernelSHAPExplainer(model, task=task, seed=seed,
                         n_cpus=1 if model_type == 'dnn' else -1)),
)




pred_contribs_map = {}
pred_y_map = {}
for expl_i, (explainer_name, explainer) in enumerate(explainer_array):
    print('Explaining model using', explainer_name)
    explainer.fit(dataset)  
    pred_contribs, y_pred = explainer.feature_contributions(
        X_subset, as_dict=True, return_predictions=True)

    
    pred_contribs_map[explainer_name] = pred_contribs
    pred_y_map[explainer_name] = y_pred




df, df_3d = gather_viz_data(
    model=model,
    dataset=dataset,
    transformer=transformer,
    true_contribs=true_contribs,
    pred_contribs_map=pred_contribs_map,
    effectwise_err_func=None,
)

col_wrap = 4

if not df.empty:
    g = sns.relplot(
        data=df,
        x='Feature Value',
        y='Contribution',
        hue='Explainer',
        
        col='Class' if task == 'classification' else 'Match',
        col_wrap=None if task == 'classification' else col_wrap,
        
        row='Match' if task == 'classification' else None,
        kind='scatter',
        x_jitter=.08,  
        alpha=.65,
        facet_kws=dict(sharex=False, sharey=False),
    )
    for ax in g.axes.flat:
        title = ax.get_title()
        ax.set_title(title.split(' = ', 1)[1])
    g.savefig(nonexistent_filename(f'contributions_grid_{model_type}.pdf'))


if not df_3d.empty:

    plt_x = 'Feature Value x'
    plt_y = 'Feature Value y'
    plt_z = 'Contribution'
    plt_hue = 'Explainer'
    plt_col = 'Match'

    df_3d_grouped = df_3d.groupby(['Class', plt_col])

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
    fig.savefig(nonexistent_filename(
        f'contributions_grid_interact_{model_type}.pdf'))

if plt.get_backend() != 'agg':
    plt.show()
