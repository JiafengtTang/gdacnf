#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, rankdata
#import util
import datasets2
import CNF as C


# In[2]:


plt_settings = dict(margin=dict(t=30, b=30, r=30),
                    font=dict(family="PTSerif", size=18),
                    legend=dict(orientation="h", bordercolor="Black", borderwidth=0.3, yanchor="bottom", y=-0.35, xanchor="center", x=0.5))


# In[3]:


def add_var_scatter_plot(fig, x, y, color, name=None, showlegend=True, **kwargs):
    """
    @param
    fig: go.Figure
    color: int, we prepare 10 colors, you can select the number 0 to 9.
    name: str, the name of plot
    """
    colors = [list(px.colors.hex_to_rgb(_hex)) for _hex in px.colors.qualitative.Plotly]
    rgb = 'rgb' + str(tuple(colors[color]))
    rgba = 'rgba' + str(tuple(colors[color] + [0.3]))  # opacity = 0.3
    mean, std = np.nanmean(y, axis=1), np.nanstd(y, ddof=1, axis=1)
    fig.add_scatter(x=x, y=mean, name=name, mode='markers+lines', line=dict(color=rgb), showlegend=showlegend, **kwargs)
    fig.add_scatter(x=x, y=mean+std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='none', **kwargs)
    fig.add_scatter(x=x, y=mean-std, mode='lines', fill="tonexty", line=dict(width=0),
                    showlegend=False, hoverinfo='none', fillcolor=rgba, **kwargs)
    return fig


def add_box_plot(fig, y, color, **kwargs):
    """
    @param
    fig: go.Figure
    color: int, we prepare 10 colors, you can select the number 0 to 9.
    name: str, the name of plot
    """
    colors = [list(px.colors.hex_to_rgb(_hex)) for _hex in px.colors.qualitative.Plotly]
    rgb = 'rgb' + str(tuple(colors[color]))
    black = 'rgb(0,0,0)'
    y = np.array(y).squeeze()
    fig.add_box(y=y, fillcolor=rgb, line=dict(color=black),  **kwargs)
    return fig


# # toy data

# ## with intermediate and TPR

# In[ ]:


model = C.build_cnf(input_dims=2, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn='moon-TPR')
model.load_state()

x_all, y_all = datasets2.make_gradual_data()
_ = x_all.pop()

x_gnr = C.get_generate_gradual_samples(model, x_all, 0.5)


# In[ ]:


colors = [list(px.colors.hex_to_rgb(_hex)) for _hex in px.colors.qualitative.Plotly]
rgb = ['rgb' + str(tuple(colors[i])) for i in [0, 3, 0, 3, 0]]
titles = ['Original, t=1', 'Generated, t=1.5', 'Original, t=2', 'Generated, t=2.5', 'Original, t=3']

fig = make_subplots(rows=1, cols=5, x_title='x1', y_title='x2', shared_xaxes=True, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.01)
for i, (x, c) in enumerate(zip(x_gnr, rgb)):
    fig.add_scatter(x=x[:, 0], y=x[:, 1], row=1, col=i+1, mode='markers', marker=dict(color=c), showlegend=False)
#x_ticks = {f'xaxis{i+1}': dict(tickmode='array', tickvals=[-1, 0, 1, 2]) for i in range(len(titles))}
fig.update_layout(font=plt_settings['font'], width=1200, height=400,)
fig.update_annotations(font=dict(size=22))
fig.write_image('rot-moon-TPR.pdf')
fig.show()


# ## with intermediate and without TPR

# In[4]:


model = C.build_cnf(input_dims=2, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn='moon')
model.load_state()

x_all, y_all = datasets2.make_gradual_data()
_ = x_all.pop()

x_gnr = C.get_generate_gradual_samples(model, x_all, 0.5)

colors = [list(px.colors.hex_to_rgb(_hex)) for _hex in px.colors.qualitative.Plotly]
rgb = ['rgb' + str(tuple(colors[i])) for i in [0, 3, 0, 3, 0]]
titles = ['Original, t=1', 'Generated, t=1.5', 'Original, t=2', 'Generated, t=2.5', 'Original, t=3']

fig = make_subplots(rows=1, cols=5, x_title='x1', y_title='x2', shared_xaxes=True, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.01)
for i, (x, c) in enumerate(zip(x_gnr, rgb)):
    fig.add_scatter(x=x[:, 0], y=x[:, 1], row=1, col=i+1, mode='markers', marker=dict(color=c), showlegend=False)
#x_ticks = {f'xaxis{i+1}': dict(tickmode='array', tickvals=[-1, 0, 1, 2]) for i in range(len(titles))}
fig.update_layout(font=plt_settings['font'], width=1200, height=400,)
fig.update_annotations(font=dict(size=22))
fig.write_image('rot-moon.pdf')
fig.show()


# ### without intermediate and with TPR

# In[14]:


model = C.build_cnf(input_dims=2, hidden_dims=64, num_hidden_layers=3, num_blocks=3, fn='moon-direct-TPR')
model.load_state()

x_all, y_all = datasets2.make_gradual_data(steps=2)
_ = x_all.pop()

x_gnr = C.get_generate_gradual_samples(model, x_all, 0.3)

colors = [list(px.colors.hex_to_rgb(_hex)) for _hex in px.colors.qualitative.Plotly]
rgb = ['rgb' + str(tuple(colors[i])) for i in [0, 3, 3, 3, 0]]
titles = ['Original, t=1', 'Generated, t=1.3', 'Generated, t=1.6', 'Generated, t=1.9', 'Original, t=2']

fig = make_subplots(rows=1, cols=5, x_title='x1', y_title='x2', shared_xaxes=True, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.01)
for i, (x, c) in enumerate(zip(x_gnr, rgb)):
    fig.add_scatter(x=x[:, 0], y=x[:, 1], row=1, col=i+1, mode='markers', marker=dict(color=c), showlegend=False)
#x_ticks = {f'xaxis{i+1}': dict(tickmode='array', tickvals=[-1, 0, 1, 2]) for i in range(len(titles))}
fig.update_layout(font=plt_settings['font'], width=1200, height=400,)
fig.update_annotations(font=dict(size=22))
fig.write_image('rot-moon-direct.pdf')
fig.show()


# # UMAP

# In[ ]:


names = {'mnist': 'Rotating MNIST',
         'portraits': 'Portraits',
         'rxrx1': 'RxRx1',
         'shift15m': 'SHIFT15M'}

fig = make_subplots(rows=1, cols=len(names), y_title='accuracy',
                    shared_xaxes=True, shared_yaxes=True,
                    subplot_titles=list(names.values()), horizontal_spacing=0.03)

for col, name in enumerate(names):
    result = pd.read_pickle(f'data_{name}.pkl')
    n_components = list(result.keys())
    accuracy = [result[k][-1] for k in n_components]
    #max_acc_ndim = n_components[np.argmax(np.mean(accuracy, axis=1))]
    fig = add_var_scatter_plot(fig, x=n_components, y=accuracy, color=0, name=name, showlegend=False, row=1, col=col+1)
xaxis_title = {f'xaxis{i+1}_title': 'number of dims' for i in range(len(names))}
fig.update_layout(**xaxis_title, **plt_settings, width=1800, height=400)
fig.update_annotations(font=dict(size=22))
fig.write_image('fitUMAP.pdf')
fig.show()


# # real data

# ## corr

# In[ ]:


settings = {'mnist': 'Rotating MNIST',
            'portraits': 'Portraits',
            'tox21c': 'Tox21: NumHDonors',
            'shift15m': 'SHIFT15M',
            'rxrx1': 'RxRx1'}

corr = []
for name, title in settings.items():
    result = pd.read_pickle(f'{name}-TPR_gdacnf.pkl')
    _, _, forward, backward = result.values()
    forward, backward = np.mean(forward, axis=1), np.mean(backward, axis=1)
    f_rank, b_rank = rankdata(forward), rankdata(backward)
    corr.append([pearsonr(forward, backward)[0]])

corr_df = pd.DataFrame(corr, index=settings.values(), columns=['peason'])
corr_df


# ## compare with baseline

# In[ ]:


titles = ['Rotating MNIST', 'Portraits', 'Tox21', 'SHIFT15M', 'RxRx1']
fig = make_subplots(rows=1, cols=len(titles), y_title='accuracy',
                    shared_xaxes=True, #shared_yaxes=True,
                    subplot_titles=titles, horizontal_spacing=0.03)


dataset = ['mnist', 'portraits', 'tox21c', 'shift15m', 'rxrx1']
#dataset = ['portraits', 'tox21a', 'shift15m', 'rxrx1']
methods = {'gdacnf': 'Ours',
           'sourceonly': 'SourceOnly',
           'gst': 'GradualSelfTrain',
           'gift-low': 'GIFT-low',
           'gift-mid': 'GIFT-mid',
           'gift-high': 'GIFT-high',
           'aux-low': 'AuxSelfTrain-low',
           'aux-mid': 'AuxSelfTrain-mid',
           'aux-high': 'AuxSelfTrain-high'}


for col, data in enumerate(dataset):
    showlegend = True if col == 0 else False
    for color, method in enumerate(methods):
        fn = f'{data}_{method}.pkl' if method != 'gdacnf' else f'{data}-TPR_{method}.pkl'
        result = pd.read_pickle(fn)
        if method == 'gdacnf':
            idx = np.nanmean(result['f'], axis=1).argmax()
            acc = result['f'][idx, :]
        else:
            acc = result['acc']
        fig = add_box_plot(fig, acc, color, row=1, col=col+1, name=methods[method], showlegend=showlegend)


xaxis_visible = {f'xaxis{i+1}_visible': False for i in range(len(dataset))}
fig.update_layout(**xaxis_visible, **plt_settings, width=1800, height=400)
fig.update_annotations(font=dict(size=22))
fig.update_layout(legend=dict(yanchor="bottom", y=-0.20, xanchor="center", x=0.5))
fig.write_image('compare_result.pdf')
fig.show()


# # appendix

# ## comparison between tox21a, tox21b and tox21c

# In[ ]:


titles = ['NumHDonors', 'RingCount', 'NHOHCount']
fig = make_subplots(rows=1, cols=len(titles), y_title='accuracy',
                    shared_xaxes=True, #shared_yaxes=True,
                    subplot_titles=titles, horizontal_spacing=0.03)


dataset = ['tox21c', 'tox21b', 'tox21a']
methods = {'gdacnf': 'Ours',
           'sourceonly': 'SourceOnly',
           'gst': 'GradualSelfTrain',
           'gift-low': 'GIFT-low',
           'gift-mid': 'GIFT-mid',
           'gift-high': 'GIFT-high',
           'aux-low': 'AuxSelfTrain-low',
           'aux-mid': 'AuxSelfTrain-mid',
           'aux-high': 'AuxSelfTrain-high'}


for col, data in enumerate(dataset):
    showlegend = True if col == 0 else False
    for color, method in enumerate(methods):
        fn = f'{data}_{method}.pkl' if method != 'gdacnf' else f'{data}-TPR_{method}.pkl'
        result = pd.read_pickle(fn)
        if method == 'gdacnf':
            idx = np.nanmean(result['f'], axis=1).argmax()
            acc = result['f'][idx,:]
        else:
            acc = result['acc']
        fig = add_box_plot(fig, acc, color, row=1, col=col+1, name=methods[method], showlegend=showlegend)


xaxis_visible = {f'xaxis{i+1}_visible': False for i in range(len(dataset))}
fig.update_layout(**xaxis_visible, **plt_settings, width=1400, height=400)
fig.update_annotations(font=dict(size=22))
fig.update_layout(legend=dict(yanchor="bottom", y=-0.20, xanchor="center", x=0.5))
fig.write_image('tox21_result.pdf')
fig.show()


# In[ ]:




