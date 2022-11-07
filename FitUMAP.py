#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import tensorflow as tf
import umap
from umap.parametric_umap import ParametricUMAP

import datasets2
import GradualSelfTrain as G


# In[ ]:


def fit_umap(x_all, y_all, **umap_kwargs) -> list:
    umap_settings = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_settings.update(umap_kwargs)
    X = np.vstack(x_all)
    X = X.reshape(X.shape[0], -1)
    # use source label as semi-superviesd UMAP
    Y_semi_supervised = [np.full(shape=y.shape[0], fill_value=-1) for y in y_all]
    Y_semi_supervised[0] = y_all[0].copy()
    Y_semi_supervised = np.hstack(Y_semi_supervised)
    # fit UMAP
    encoder = umap.UMAP(random_state=1234, **umap_settings)
    Z = encoder.fit_transform(X, Y_semi_supervised)
    z_idx = np.cumsum([i.shape[0] for i in x_all])
    z_all = np.vsplit(Z, z_idx)[:-1]
    return z_all, encoder


def fit_parametric_umap(x_all, y_all, **umap_kwargs):
    umap_settings = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_settings.update(umap_kwargs)
    X = np.vstack(x_all)
    X = X.reshape(X.shape[0], -1)
    # use source label as semi-superviesd UMAP
    Y_semi_supervised = [np.full(shape=y.shape[0], fill_value=-1) for y in y_all]
    Y_semi_supervised[0] = y_all[0].copy()
    Y_semi_supervised = np.hstack(Y_semi_supervised)
    # fit UMAP
    embedder = ParametricUMAP(parametric_embedding=True, verbose=True,
                              random_state=1234,
                              **umap_settings)
    Z = embedder.fit_transform(X, Y_semi_supervised)
    z_idx = np.cumsum([i.shape[0] for i in x_all])
    z_all = np.vsplit(Z, z_idx)[:-1]
    return z_all, embedder


# In[ ]:


settings = {
            'mnist': datasets2.load_RotatedMNIST2,
            'portraits': datasets2.load_Portraits,
            'rxrx1': datasets2.load_RxRx1,
            'shift15m': datasets2.load_shift15m,
           }

components = np.arange(2, 10)
rep = 20

random.seed(1111)
np.random.seed(2222)
tf.random.set_seed(3333)


# In[ ]:


warnings.simplefilter('ignore')

for name in settings.keys():
    print(f'\n{name}\n')

    results = {}
    func = settings[name]
    x_all, y_all = func()
    for c in components:
        z_all, _ = fit_umap(x_all, y_all, n_components=c)
        # eval embedding by using source label
        accuracy = []
        for r in range(rep):
            models, _ = G.SourceOnly(z_all, y_all)
            acc = G.calc_accuracy(models[-1], z_all[0], y_all[0])
            accuracy.append(acc)

        results[c] = (z_all, y_all, accuracy)
        pd.to_pickle(results, f'data_{name}.pkl')

# make pkl file for rot-moon and Tox21
x_all, y_all = datasets2.make_gradual_data()
obj = {2: (x_all, y_all, None)}
pd.to_pickle(obj, 'data_moon.pkl')

x_all, y_all = datasets2.make_gradual_data(steps=2)
obj = {2: (x_all, y_all, None)}
pd.to_pickle(obj, 'data_moon-direct.pkl')

for suffix, domain in zip(['a', 'b', 'c'], ['NHOH', 'RingCount', 'NumHDonors']):
    x_all, y_all = datasets2.load_Tox21(domain=domain)
    obj = {5: (x_all, y_all, None)}
    pd.to_pickle(obj, f'data_tox21{suffix}.pkl')

