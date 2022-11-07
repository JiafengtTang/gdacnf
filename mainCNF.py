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
# my module
import CNF as C

# dataset name: z_dim, given_index
settings = {
            'moon': (2, None),
            'moon-direct': (2, None),
            'mnist': (4, [0, 14, 28]),
            'portraits': (8, None),
            'tox21a': (5, None),
            'tox21b': (5, None),
            'tox21c': (5, None),
            'rxrx1': (9, None),
            'shift15m': (6, None),
           }


if __name__ == '__main__':
    # args are dataset, TPR flag and GPU No.
    # command example, python mainCNF.py mnist 1 0
    epochs = 8000
    name, tpr_flag, C.util.gpu_no = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    filename = f'{name}-TPR' if tpr_flag else f'{name}'
    print(name)

    random.seed(1111)
    np.random.seed(2222)
    torch.manual_seed(3333)

    z_dim, given_domain = settings[name]
    # umap involve eval data
    z_all, y_all, _ = pd.read_pickle(f'data_{name}.pkl')[z_dim]
    # in cnf, eval data not need
    _ = z_all.pop()
    if given_domain is None:
        z_subset = z_all
    else:
        z_subset = [z_all[i].copy() for i in given_domain]
    #divergence_fn = "approximate" if name == 'mnist' else "brute_force"
    model = C.build_cnf(input_dims=z_dim, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn=filename)
    model, lh = C.train_cnf(model, z_subset, epochs, tpr_flag)
