# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:.conda-torch-env]
#     language: python
#     name: conda-env-.conda-torch-env-py
# ---

import sys
import random
import torch
import numpy as np
import pandas as pd
import GradualSelfTrain as G

# +
# settings
rep = 20

# dataset name: z_dim, given_index
settings = {
            'moon': (2, None),
            'mnist': (4, [0, 14, 28]),
            'portraits': (8, None),
            'tox21a': (5, None),
            'tox21b': (5, None),
            'tox21c': (5, None),
            'rxrx1': (9, None),
            'shift15m': (6, None),
           }

# key: train function, param
methods = {'sourceonly': (G.SourceOnly, None),
           'targetonly': (G.TargetOnly, None),
           'gst': (G.GradualSelfTrain, None),
           'gift-low': (G.GIFT, 10),
           'gift-mid': (G.GIFT, 20),
           'gift-high': (G.GIFT, 30),
           'aux-low': (G.AuxSelfTrain, 10),
           'aux-mid': (G.AuxSelfTrain, 20),
           'aux-high': (G.AuxSelfTrain, 30)}
# -

if __name__ == '__main__':
    # args are dataset, method and GPU No.
    # python mainBaseline.py mnist gift-low 1
    key, m, G.util.gpu_no = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print(f'{key}_{m}')
    # load data
    z_dim, given_domain = settings[key]
    z_all, y_all, _ = pd.read_pickle(f'data_{key}.pkl')[z_dim]
    z_eval, y_eval = z_all.pop(), y_all.pop()  # umap involve eval data
    if given_domain is None:
        z_subset = z_all
        y_subset = y_all
    else:
        z_subset = [z_all[i].copy() for i in given_domain]
        y_subset = [y_all[i].copy() for i in given_domain]

    func, param = methods[m]
    res = np.full(shape=(1, rep), fill_value=np.nan)
    for r in range(rep):
        random.seed(r)
        np.random.seed(r)
        torch.manual_seed(r)

        if param is None:
            all_model, _ = func(z_subset, y_subset)
        else:
            all_model, _ = func(z_subset, y_subset, param)
        res[0, r] = G.calc_accuracy(all_model[-1], z_eval, y_eval)
        pd.to_pickle({'acc': res}, f'{key}_{m}.pkl')
