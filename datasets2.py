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

import json
import gzip
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
from scipy import ndimage
from rdkit import RDLogger
from rdkit.Chem import Descriptors, PandasTools
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
import torchvision.transforms as tforms
import torch
from torch.utils.data import DataLoader

# +
data_dir = Path('/home/')


def load_RotatedMNIST2(start=0, end=60, num_inter_domain=27, num_domain_samples=2000):
    """
    @param
    start, end: int, rotate angles
    num_inter_domain: int, how many intermediate domains needed
    num_inter_samples: set the same sample size in all domains (source, inter, target, eval)
    """
    global data_dir
    np.random.seed(1234)
    # load MNIST
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # rotated mnist does not need test data
        dataset = MNIST(data_dir, train=True, download=True)
    x = np.array(dataset.data).astype(np.float32) / 255
    y = np.array(dataset.targets)
    # set angles
    angles = np.linspace(start, end, num_inter_domain+2)
    angles = np.append(angles, end)
    # set sample size and index
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    each_domain_samples = np.full(shape=(num_inter_domain+3), fill_value=num_domain_samples)  # source + inter + target +eval
    split_index = np.split(index, np.cumsum(each_domain_samples))
    # rotate
    x_all, y_all = list(), list()
    for idx, angle in zip(split_index, angles):
        #rotated_x = np.array([ndimage.rotate(i, np.random.normal(loc=angle, scale=5), reshape=False) for i in x[idx]])
        rotated_x = np.array([ndimage.rotate(i, angle, reshape=False) for i in x[idx]])
        x_all.append(rotated_x.reshape(-1, 1, 28, 28))
        y_all.append(y[idx])
    return x_all, y_all


def make_split_data(df: pd.DataFrame, target: str, num_inter_domain: int, num_domain_samples: dict):
    """ use for Portraits, Gas Sensor, Cover Type """
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.drop(target, axis=1).loc[idx].values
        y = df.loc[idx, target].values
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def shuffle_target_and_eval(x_all: list, y_all: list):
    """ use for Portraits, Gas Sensor, Cover Type """
    tx, ty = x_all[-2].copy(), y_all[-2].copy()
    ex, ey = x_all[-1].copy(), y_all[-1].copy()
    marge_x = np.vstack([tx, ex])
    marge_y = np.hstack([ty, ey])
    idx = np.arange(marge_x.shape[0])
    np.random.seed(1234)
    np.random.shuffle(idx)
    t_idx, e_idx = idx[:tx.shape[0]], idx[tx.shape[0]:]
    x_all[-2], y_all[-2] = marge_x[t_idx], marge_y[t_idx]
    x_all[-1], y_all[-1] = marge_x[e_idx], marge_y[e_idx]
    return x_all, y_all


def load_Portraits(num_inter_domain=6, num_domain_samples='default', use_domain_index=[0, 3, 7, 8], return_df=False):
    """
    @param
    num_inter_domain: inter domain data will be vsplit by this param
    num_domain_samles: number of samples in each domain.

    @memo
    image shape will be change, (N, height, width) -> (N, 1, height, width)
    https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0

    @ Kumar's setting
    In Kumar's setting, his last intermediate domain equal to our target domain
    """
    global data_dir
    if num_domain_samples == 'default':
        num_domain_samples = {'source': 2000, 'inter': 12000, 'target': 2000, 'eval': 2000}

    def read_path(sex: int):
        p = 'portraits/F' if sex == 1 else 'portraits/M'
        p = Path(data_dir) / p
        p_list = list(p.glob("*.png"))
        data_frame = pd.DataFrame({'img_path': p_list})
        data_frame['sex'] = sex
        return data_frame

    def convert_portraits(p: Path):
        # read, gray scale, resize
        img = Image.open(p).convert('L').resize((32,32), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32) / 255
        return img

    # prepare portraits image, female as 1, male as 0
    df = pd.concat([read_path(1), read_path(0)]).reset_index(drop=True)
    df['year'] = df['img_path'].apply(lambda p: p.stem.split('_')[0]).astype(int)
    if return_df:
        df['decade'] = df['year'].apply(lambda y: int(str(y)[:3]+'0'))
        return df
    df = df.sort_values(by='year').reset_index(drop=True).drop('year', axis=1)

    # split to each domain
    x_all, y_all = make_split_data(df, 'sex', num_inter_domain, num_domain_samples)
    x_all, y_all = shuffle_target_and_eval(x_all, y_all)
    x_all = [x_all[i].copy() for i in use_domain_index]
    y_all = [y_all[i].copy() for i in use_domain_index]
    for i, domain in enumerate(x_all):
        domain = np.array([convert_portraits(x) for x in domain.flatten()])
        x_all[i] = domain.reshape(-1, 1, 32, 32)
    return x_all, y_all


def make_gradual_data(steps=3, n_samples=2000, start=0, end=90):
    """
    @param
    steps: int, how gradual is it
    n_samples: int, how many samples, each domains
    start: int, param of shift
    end: int, param of shift
    """
    x, y = make_moons(n_samples=n_samples, random_state=8, noise=0.05)
    shifts = np.linspace(start, end, steps)
    x_all, y_all = list(), list()
    for shift in shifts:
        x_all.append(_convert_moon(x, shift))
        y_all.append(y)
        # for eval data
        if shift == shifts[-1]:
            x_all.append(_convert_moon(x, shift))
            y_all.append(y)
    return x_all, y_all


def _convert_moon(x: np.ndarray, shift: int) -> np.ndarray:
    x_copy = x.copy()
    rad = np.deg2rad(shift)
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)],
                           [-np.sin(rad), np.cos(rad)]])
    rot_x = x_copy @ rot_matrix
    return rot_x.astype(np.float32)


def load_Tox21(domain: str, eval_size: int = 500, seed: int=1234):
    """
    @param
    domain: str, the indicator which divide the domain, NHOH/RingCount/NumHDonors
    eval_size: target domain spilit to target and eval dataset
    seed: random seed for train_test_split

    @memo
    We count the number of substituents of the compound and consider the number of substituents as a domain.
    NHOHCount 0 -> source, 1 -> inter, 2 -> target and eval
    """
    df = pd.read_csv(data_dir / 'tox21.csv.gz')
    # We consider compounds as toxic that the compound shows a positive reaction for any of the tests.
    df['ToxSum'] = df.iloc[:, :12].sum(axis=1, skipna=True)
    df['y'] = df['ToxSum'].apply(lambda s: 1 if s >= 1 else 0)
    # add Mol object
    RDLogger.DisableLog('rdApp.*')
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')
    df['NHOH'] = df['ROMol'].apply(Descriptors.NHOHCount)
    df['RingCount'] = df['ROMol'].apply(Descriptors.RingCount)
    df['NumHDonors'] = df['ROMol'].apply(Descriptors.NumHDonors)
    # These features are reported in the previous work.
    features = ['Chi0v', 'Chi1n', 'Kappa1', 'HallKierAlpha', 'TPSA']
    for name, function in Descriptors.descList:
        if name in features:
            df[name] = df['ROMol'].apply(function)
    # Scaling
    x = df.loc[:, features].values
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = df['y'].values
    x_all, y_all = [], []
    for i in [0, 1, 2]:  # source, inter, target
        if domain == 'NHOH':
            idx = df.query('NHOH==@i').index
        elif domain == 'RingCount':
            idx = df.query('RingCount==@i').index
        elif domain == 'NumHDonors':
            idx = df.query('NumHDonors==@i').index
        x_all.append(x[idx])
        y_all.append(y[idx])
    # target domain split to target and eval
    x_target, x_eval, y_target, y_eval = train_test_split(x_all[-1], y_all[-1], test_size=eval_size,
                                                          stratify=y_all[-1], random_state=seed)
    _, _ = x_all.pop(), y_all.pop()
    x_all += [x_target, x_eval]
    y_all += [y_target, y_eval]
    return x_all, y_all


def load_RxRx1(eval_size: int=3000, seed: int=1234):
    """
    @param
    eval_size: target domain spilit to target and eval dataset
    seed: random seed for train_test_split

    @memo
    We estimate the cell type by using the information from images.
    number of experiment 1 -> source, 2 -> inter, 3 -> target and eval
    """
    rxrx1_dir = data_dir / 'rxrx1_v1.0'
    meta_df = pd.read_csv(rxrx1_dir / 'metadata.csv')
    meta_df['cell_type_id'] = meta_df['cell_type'].astype('category').cat.codes
    meta_df['num_experiment'] = meta_df['experiment'].apply(lambda s: int(s.split('-')[1]))
    # add path of images
    meta_df['img_path'] = rxrx1_dir / 'images' / (meta_df['experiment'] + "/Plate" + meta_df['plate'].astype(str) \
                           + "/" + meta_df['well'] + "_s" + meta_df['site'].astype(str) + ".png")
    x_all, y_all = [], []
    for nx in [1, 2, 3]:  # source, inter, target
        x = []
        idx = meta_df.query('num_experiment==@nx').index.values
        for i in idx:
            # The size of original image is 256 * 256
            img = Image.open(meta_df.loc[i, 'img_path']).resize((32,32), Image.ANTIALIAS)
            img = np.array(img, dtype=np.float32) / 255
            x.append(img.flatten())
        x = np.array(x)
        y = meta_df.loc[idx, 'cell_type_id'].values
        x_all.append(x)
        y_all.append(y)
    # target domain split to target and eval
    x_target, x_eval, y_target, y_eval = train_test_split(x_all[-1], y_all[-1], test_size=eval_size,
                                                           stratify=y_all[-1], random_state=seed)
    _, _ = x_all.pop(), y_all.pop()
    x_all += [x_target, x_eval]
    y_all += [y_target, y_eval]
    return x_all, y_all


def load_shift15m(sample_size: int=5000, seed: int=1234):
    """
    @param
    sample_size: sampling size of each year
    seed: random seed for train_test_split

    @memo
    2010&2011 -> source, 2015 -> inter, 2020 -> target
    """
    shift_dir = data_dir / 'shift15m/data'
    item_catalog = pd.read_csv(shift_dir/'item_catalog.txt', header=None, sep=" ",
                               names=["item_id", "category", "subcategory", "year"])
    item_catalog['category_id'] = item_catalog['category'].astype('category').cat.codes
    item_catalog['year'] = item_catalog['year'].replace(2010, 2011)  # merge 2010 and 2011
    # get indices of each domain
    idx_all = []
    #years = sorted(item_catalog['year'].unique())
    for qyear in [2011, 2015, 2020]:
        subset = item_catalog.query('year==@qyear').copy()
        idx = subset['item_id'].index.values
        y = subset['category'].values
        sample_idx, _ = train_test_split(idx, train_size=sample_size, stratify=y, random_state=seed)
        idx_all.append(sample_idx)
        # for eval data
        if qyear == 2020:
            sample_idx, _ = train_test_split(idx, train_size=sample_size, stratify=y, random_state=seed*2)
            idx_all.append(sample_idx)
    # load data
    x_all, y_all = [], []
    for i in idx_all:
        x = []
        for j in item_catalog.loc[i]['item_id'].tolist():
            path = (shift_dir / 'features') / f'{j}.json.gz'
            with gzip.open(path, "r") as f:
                feature = np.array(json.load(f), dtype=np.float32)
                x.append(feature)
        x_all.append(np.array(x))
        y_all.append(item_catalog.loc[i]['category_id'].values)
    return x_all, y_all