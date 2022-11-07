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
import GradualSelfTrain as G
import CNF as C

# +
rep = 20
alp_candidate = [0.1, 0.2, 0.3, 0.5, 0.8, 1]

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

# + tags=[]
if __name__ == '__main__':
    # args are dataset, TPR flag and GPU No.
    # command example, python mainGST.py mnist 1 0
    name, tpr_flag, gpu_no = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    C.util.gpu_no = G.util.gpu_no = gpu_no
    # args are dataset, TPR flag
    # command example, mainGST.py mnist 1
    name, tpr_flag = sys.argv[1], int(sys.argv[2])
    filename = f'{name}-TPR' if tpr_flag else f'{name}'
    print(filename)

    z_dim, given_domain = settings[name]

    z_all, y_all, _ = pd.read_pickle(f'data_{name}.pkl')[z_dim]
    z_eval, y_eval = z_all.pop(), y_all.pop()
    if given_domain is None:
        z_subset = z_all
        y_subset = y_all
    else:
        z_subset = [z_all[i].copy() for i in given_domain]
        y_subset = [y_all[i].copy() for i in given_domain]

    forward = np.full(shape=(len(alp_candidate), rep), fill_value=np.nan)
    backward = np.full_like(forward, fill_value=np.nan)
    num_domain = []  # source + inter + target

    for i, alpha in enumerate(alp_candidate):

        model = C.build_cnf(input_dims=z_dim, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn=filename)
        model.load_state()

        z_gnr = C.get_generate_gradual_samples(model, z_subset, alpha)
        num_domain.append(len(z_gnr))

        for r in range(rep):
            random.seed(r)
            np.random.seed(r)
            torch.manual_seed(r)

            forward_models, backward_models = G.CycleConsistencySelfTrain(z_gnr, y_subset)
            forward[i, r] = G.calc_accuracy(forward_models[-1], z_eval, y_eval)
            backward[i, r] = G.calc_accuracy(backward_models[-1], z_subset[0], y_subset[0])

        obj = {'alpha': alp_candidate, 'T': np.array(num_domain), 'f': forward, 'b': backward}
        pd.to_pickle(obj, f'{filename}_gdacnf.pkl')
# -

