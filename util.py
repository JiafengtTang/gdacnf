#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
from torch.utils.data import TensorDataset


def torch_to(*args):
    global gpu_no
    if 'gpu_no' not in globals():
        gpu_no = 0
    device = torch.device(f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    return [arg.to(device) for arg in args] if len(args) > 1 else args[0].to(device)


def preprocess_input(*args) -> TensorDataset:
    all_tensors = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            # this is label data
            if arg.ndim == 1:
                arg = torch.tensor(np.array(arg).astype(int))
            # this is feature data
            else:
                arg = torch.tensor(np.array(arg).astype(np.float32))
        all_tensors.append(arg)
    dataset = TensorDataset(*all_tensors)
    return dataset


def get_expand_range(series:pd.Series, ratio:int=10) -> list:
    """ use for plot """
    d_min, d_max = series.min(), series.max()
    upper = d_max + (d_max * ratio / 100) if d_max > 0 else d_max - (d_max * ratio / 100)
    lower = d_min - (d_min * ratio / 100) if d_min > 0 else d_min + (d_min * ratio / 100)
    return [lower, upper]


def plot_gradual(x_all:list, y_all:list=None):
    # make data frame
    df = []
    for i, x in enumerate(x_all):
        _df = pd.DataFrame(x, columns=['x1', 'x2'])
        _df['frame'] = i
        if y_all != None:
            _df['y'] = y_all[i].astype(str)
        df.append(_df)
    df = pd.concat(df)
    # plot
    color = 'y' if y_all != None else None
    fig = px.scatter(data_frame=df, x='x1', y='x2', animation_frame='frame', color=color,
                     range_x=get_expand_range(df['x1']), range_y=get_expand_range(df['x2']), width=600, height=600)
    return fig


def visualize_predict(model, x, y, mesh_points=50) -> go.Figure:
    """ 2d and 2-class data only """
    x1_min, x1_max = get_expand_range(x[:,0])
    x2_min, x2_max = get_expand_range(x[:,1])
    x1range = np.linspace(x1_min, x1_max, mesh_points)
    x2range = np.linspace(x2_min, x2_max, mesh_points)
    x1x1, x2x2 = np.meshgrid(x1range, x2range)
    # estimate prob for all mesh points
    mesh = np.c_[x1x1.ravel(), x2x2.ravel()]
    dataset = preprocess_input(mesh)
    model, dataset = torch_to(model, dataset.tensors[0])
    dataset = dataset.squeeze()
    with torch.no_grad():
        logits = model(dataset)
        logits = torch.nn.functional.softmax(logits, dim=1)
        #z = np.array(logits)[:,1]
        z = logits.cpu().detach().numpy()[:,1]
    z = z.reshape(x1x1.shape)
    # plot
    yA, yB = np.unique(y)
    fig = go.Figure(data=[go.Scatter(x=x[y==yA, 0], y=x[y==yA, 1], mode='markers')])
    fig.add_scatter(x=x[y==yB, 0], y=x[y==yB, 1], mode='markers')
    fig.update_layout(width=600, height=500, xaxis_title='x1', yaxis_title='x2',
                      margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    fig.add_trace(go.Contour(x=x1range, y=x2range, z=z, showscale=False, colorscale=['blue', 'white', 'red'], opacity=0.3))
    return fig


def plot_loss_history(path: str):
    lh = pd.read_pickle(path)
    now = np.sum(~np.isnan(lh[0,:]))
    total = lh[0,:].shape[0]
    fig = px.scatter(title=f'{now}/{total} epochs')
    for i in range(lh.shape[0]):
        fig.add_scatter(y=lh[i,:], mode='markers', name=f'time={i+1}')
    fig.update_layout(margin=dict(t=30, b=30), xaxis_title='epochs', yaxis_title='loss')
    fig.show()
    return lh


def rounded_statistics(array, ndigits=3):
    m, s = round(np.mean(array), ndigits), round(np.std(array), ndigits)
    return '{}±{}'.format(m, s)
