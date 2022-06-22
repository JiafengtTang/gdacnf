import sys
import numpy as np
import pandas as pd
# my module
import util
import datasets2
import FFJORD2 as cnf

# dataset name: given domain, z_dim
settings = {
            'mnist': ([0, 10, 21], 5),
            'portraits': ([0, 3, 7], 3),
            'gas': ([0, 1, 2], 3),
           }

if __name__ == '__main__':

    epochs = 8000
    name = sys.argv[1]
    print(name)

    given_domain, z_dim = settings[name]
    # umap involve eval data
    z_all, y_all, _ = pd.read_pickle(f'umap_{name}.pkl')[z_dim]
    # cnf not involve eval data and some intermediate data
    z_subset = [z_all[i].copy() for i in given_domain]
    model = cnf.build_cnf(input_dims=z_dim, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn=name)
    model, lh = cnf.train_cnf(model, z_subset, epochs)
