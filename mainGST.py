import sys
import numpy as np
import pandas as pd
# my module
import util
import GradualSelfTrain as gst
import FFJORD2 as cnf

# +
rep = 5
alp_candidate = [0.05, 0.1, 0.2, 0.3, 0.5, 1]

# dataset name: given domain, z_dim
settings = {
            'mnist': ([0, 10, 21], 5),
            'portraits': ([0, 3, 7], 3),
           }

# + tags=[]
if __name__ == '__main__':

    name = sys.argv[1]
    print(name)

    given_domain, z_dim = settings[name]
    # umap involve eval data
    z_all, y_all, _ = pd.read_pickle(f'umap_{name}.pkl')[z_dim]
    z_eval, y_eval = z_all.pop(), y_all.pop()
    z_subset = [z_all[i].copy() for i in given_domain]
    y_subset = [y_all[i].copy() for i in given_domain]

    model = cnf.build_cnf(input_dims=z_dim, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn=name)
    model.load_state()

    forward = np.full(shape=(len(alp_candidate), rep), fill_value=np.nan)
    backward = np.full_like(forward, fill_value=np.nan)
    for i, alpha in enumerate(alp_candidate):
        z_gnr = cnf.get_generate_gradual_samples(model, z_subset, alpha)
        for j in range(rep):
            _, _, f_acc, b_acc = gst.cycle_consistency_self_train(z_gnr, y_subset, z_eval, y_eval)
            forward[i, j] = f_acc
            backward[i, j] = b_acc

    s_acc, t_acc = gst.eval_baseline(z_all, y_all, z_eval, y_eval, rep)
    obj = {'source': s_acc, 'target': t_acc, 'alpha': alp_candidate, 'f': forward, 'b': backward}
    pd.to_pickle(obj, f'gst_{name}.pkl')
