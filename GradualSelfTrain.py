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

import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# my module
import util


# +
class MLP(nn.Module):
    def __init__(self, num_labels, input_dim, hidden_dim):
        super(MLP, self).__init__()
        """ in the case of GIFT and two-moon data, nn.Linear is only one """
        """ in the case of Cover Type, hidden_dim=32 is recommend """
        self.num_labels = num_labels
        # tabular
        if isinstance(input_dim, int):
            self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.BatchNorm1d(num_features=hidden_dim))
            self.pred = nn.Linear(hidden_dim, num_labels)
        # image, input_dim -> ex. (28, 28)
        else:
            num_conv = 3
            conv_dim = np.rint(np.array(input_dim) / 2**num_conv)
            latent_dim = int(conv_dim[0] * conv_dim[1] * hidden_dim)
            conv_settings = dict(kernel_size=5, stride=2, padding=2)
            self.fc = nn.Sequential(nn.Conv2d(1, hidden_dim, **conv_settings), nn.ReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, **conv_settings), nn.ReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, **conv_settings), nn.ReLU(),
                                    nn.Dropout2d(p=0.5),
                                    nn.BatchNorm2d(num_features=hidden_dim),
                                    nn.Flatten())
            self.pred = nn.Linear(latent_dim, num_labels)

    def forward(self, x):
        feature = self.fc(x)
        pred_y = self.pred(feature)
        return pred_y


class AuxiliaryModel(MLP):
    """
    Zhang proposed
    Gradual Domain Adaptation via Self-Training of Auxiliary Models
    https://arxiv.org/abs/2106.09890
    https://github.com/YBZh/AuxSelfTrain
    """

    def get_prediction_with_uniform_prior(self, soft_prediction):
        soft_prediction_uniform = soft_prediction / soft_prediction.sum(0, keepdim=True).pow(0.5)
        soft_prediction_uniform /= soft_prediction_uniform.sum(1, keepdim=True)
        return soft_prediction_uniform

    def classifier_prediction(self, x_source):
        with torch.no_grad():
            pred_network = self.forward(x_source)
            pred_network = nn.functional.softmax(pred_network, dim=1)
        pred_network = self.get_prediction_with_uniform_prior(pred_network)
        pseudo_y = pred_network.argmax(dim=1)
        return pred_network, pseudo_y

    def ensemble_prediction(self, x_source, y_source, x_target):
        """ use only for self train """
        pred_network, _, = self.classifier_prediction(x_target)
        pred_kmeans = self.get_labels_from_kmeans(x_source, y_source, x_target)
        pred_lp = self.get_labels_from_lp(x_source, y_source, x_target)
        pred_kmeans = self.get_prediction_with_uniform_prior(pred_kmeans)
        pred_lp = self.get_prediction_with_uniform_prior(pred_lp)
        pred_ensemble = (pred_network + pred_kmeans + pred_lp) / 3
        pseudo_y = pred_ensemble.argmax(dim=1)
        return pred_ensemble, pseudo_y

    def get_labels_from_kmeans(self, x_source, y_source, x_target):
        with torch.no_grad():
            z_source = self.fc(x_source)
            z_target = self.fc(x_target)
        z_source_array, y_source_array, z_target_array = z_source.numpy(), y_source.numpy(), z_target.numpy()
        init = np.vstack([z_source_array[y_source_array==i].mean(axis=0) for i in np.unique(y_source_array)])
        kmeans = KMeans(n_clusters=self.num_labels, init=init, n_init=1, random_state=0).fit(z_target_array)
        centers = kmeans.cluster_centers_  # num_category * feature_dim
        centers_tensor = torch.from_numpy(centers)
        centers_tensor_unsq = torch.unsqueeze(centers_tensor, 0)
        target_u_feature_unsq = torch.unsqueeze(z_target, 1)
        L2_dis = ((target_u_feature_unsq - centers_tensor_unsq)**2).mean(2)
        soft_label_kmeans = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1)
        return soft_label_kmeans

    def get_labels_from_lp(self, x_source, y_source, x_target):
        """ label propagation """
        graphk = 20
        alpha = 0.75
        with torch.no_grad():
            labeled_features = self.fc(x_source)
            unlabeled_features = self.fc(x_target)
        labeled_onehot_gt = nn.functional.one_hot(y_source, num_classes=self.num_labels)

        num_labeled = labeled_features.size(0)
        if num_labeled > 100000:
            print('too many labeled data, randomly select a subset')
            indices = torch.randperm(num_labeled)[:10000]
            labeled_features = labeled_features[indices]
            labeled_onehot_gt = labeled_onehot_gt[indices]
            num_labeled = 10000

        num_unlabeled = unlabeled_features.size(0)
        num_all = num_unlabeled + num_labeled
        all_features = torch.cat((labeled_features, unlabeled_features), dim=0)
        unlabeled_zero_gt = torch.zeros(num_unlabeled, self.num_labels)
        all_gt = torch.cat((labeled_onehot_gt, unlabeled_zero_gt), dim=0)
        ### calculate the affinity matrix
        all_features = nn.functional.normalize(all_features, dim=1, p=2)
        weight = torch.matmul(all_features, all_features.transpose(0, 1))
        weight[weight < 0] = 0
        values, indexes = torch.topk(weight, graphk)
        weight[weight < values[:, -1].view(-1, 1)] = 0
        weight = weight + weight.transpose(0, 1)
        weight.diagonal(0).fill_(0)  ## change the diagonal elements with inplace operation.
        D = weight.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, num_all)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(num_all, 1)
        S = D1 * weight * D2  ############ same with D3 = torch.diag(D_sqrt_inv)  S = torch.matmul(torch.matmul(D3, weight), D3)
        pred_all = torch.matmul(torch.inverse(torch.eye(num_all) - alpha * S + 1e-8), all_gt)
        del weight
        pred_unl = pred_all[num_labeled:, :]
        #### add a fix value
        min_value = torch.min(pred_unl, 1)[0]
        min_value[min_value > 0] = 0
        pred_unl = pred_unl - min_value.view(-1, 1)
        pred_unl = pred_unl / pred_unl.sum(1).view(-1, 1)
        soft_label_lp = pred_unl
        return soft_label_lp


# +
def _check_input_dim(x):
    if np.ndim(x) == 4:
        input_dim = x.shape[-2:]
    else:
        input_dim = x.shape[1]
    return input_dim


def get_pseudo_y(model: nn.Module, x: torch.Tensor, confidence_q: float=0.1, GIFT: bool=False) -> (np.ndarray, np.ndarray):
    """ remove less confidence sample """
    dataset = util.preprocess_input(x)
    model, _x = util.torch_to(model, dataset.tensors[0].squeeze(0))
    with torch.no_grad():
        logits = model(_x) if not GIFT else model.pred(_x)
        logits = nn.functional.softmax(logits, dim=1)
        confidence = np.array(torch.Tensor.cpu(logits.amax(dim=1) - logits.amin(dim=1)))
        alpha = np.quantile(confidence, confidence_q)
        conf_index = np.argwhere(confidence >= alpha)[:,0]
        pseudo_y = logits.argmax(dim=1)
    return pseudo_y.detach().cpu().numpy(), conf_index


def calc_accuracy(model, x, y):
    dataset = util.preprocess_input(x)
    with torch.no_grad():
        model, _x = util.torch_to(model, dataset.tensors[0].squeeze(0))
        pred = model(_x)
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.array(torch.Tensor.cpu(pred.argmax(dim=1)))
    return accuracy_score(y, pred.squeeze())


def train_classifier(clf, x, y, n_epochs=100, weight_decay=1e-3, GIFT=False):
    model = deepcopy(clf)
    model = util.torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_f = nn.CrossEntropyLoss()
    batch_size = 128 #1024 #128
    dataset = util.preprocess_input(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loss_history = []
    for e in range(n_epochs):
        running_loss = 0
        for x_sample, y_sample in train_loader:
            x_sample, y_sample = util.torch_to(x_sample, y_sample)
            optimizer.zero_grad()
            y_pred = model(x_sample) if not GIFT else model.pred(x_sample)
            loss = loss_f(y_pred, y_sample)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / batch_size
        loss_history.append(running_loss)
    return model, loss_history

# +
def SourceOnly(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3) -> (list, list):
    input_dim = _check_input_dim(x_all[0])
    model = MLP(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    model, loss_history = train_classifier(model, x_all[0], y_all[0], n_epochs, weight_decay)
    return [model], loss_history


def TargetOnly(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3) -> (list, list):
    input_dim = _check_input_dim(x_all[-1])
    model = MLP(num_labels=np.unique(y_all[-1]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    model, loss_history = train_classifier(model, x_all[-1], y_all[-1], n_epochs, weight_decay)
    return [model], loss_history


def GradualSelfTrain(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3) -> (list, None):
    input_dim = _check_input_dim(x_all[0])
    model = MLP(num_labels=np.unique(np.hstack(y_all)).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    student_model = deepcopy(model)
    teacher_model = deepcopy(model)
    student_model, loss_history = train_classifier(student_model, x_all[0], y_all[0], n_epochs, weight_decay)
    all_model = [student_model]
    for j, x in enumerate(tqdm(x_all[1:])):
        teacher_model.load_state_dict(student_model.state_dict())
        pseudo_y, conf_index = get_pseudo_y(teacher_model, x)
        student_model, loss_history = train_classifier(student_model, x[conf_index], pseudo_y[conf_index], n_epochs, weight_decay)
        all_model.append(student_model)
    return all_model, None


def CycleConsistencySelfTrain(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3) -> (list, list):
    # train forward
    forward_models, _ = GradualSelfTrain(x_all, y_all, hidden_dim, n_epochs, weight_decay)
    # train backward with pseudo label
    x_all_inv, y_all_inv = deepcopy(x_all[::-1]), deepcopy(y_all[::-1])
    pseudo_y, conf_index = get_pseudo_y(forward_models[-1], x_all_inv[0])
    x_all_inv[0] = x_all_inv[0][conf_index]
    y_all_inv[0] = pseudo_y[conf_index]
    backward_models, _ = GradualSelfTrain(x_all_inv, y_all_inv, hidden_dim, n_epochs, weight_decay)
    return forward_models, backward_models


def AuxSelfTrain(x_all, y_all, num_inter, hidden_dim=32, n_epochs=100, weight_decay=1e-3) -> (list, None):
    """
    Zhang proposed
    Gradual Domain Adaptation via Self-Training of Auxiliary Models
    https://arxiv.org/abs/2106.09890
    https://github.com/YBZh/AuxSelfTrain
    @param
    num_inter: int, control the number of steps for adaptation
    """
    input_dim = _check_input_dim(x_all[0])
    model = AuxiliaryModel(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)

    x_source, y_source, x_target = x_all[0].copy(), y_all[0].copy(), np.vstack(x_all[1:]).copy()
    num_source = x_source.shape[0]
    num_target = x_target.shape[0]
    num_labels = np.unique(y_source).size

    def get_index_each_label(num_labels: int, num_sample: int, pred_soft: torch.Tensor, pseudo_y: torch.Tensor):
        conf_index = []
        for l in range(num_labels):
            idx = np.arange(pseudo_y.numpy().shape[0])
            l_idx = idx[pseudo_y == l]
            l_idx_sorted = np.argsort(pred_soft.amax(dim=1)[l_idx].numpy())[::-1]
            top = num_sample // num_labels
            l_idx = l_idx[l_idx_sorted[:top]]
            conf_index.append(l_idx)
        return np.hstack(conf_index)

    model, _ = train_classifier(model, x_source, y_source, n_epochs, weight_decay)
    all_model = [model]
    for m in range(1, num_inter):
        top_s = int(((num_inter - m - 1) * num_source) / num_inter)
        top_t = int(((m + 1) * num_target) / num_inter)
        if m == 1:
            x_input, y_input = torch.tensor(x_source).float(), torch.tensor(y_source).long()
        else:
            x_input, y_input = torch.tensor(x_inter).float(), torch.tensor(y_inter).long()
        model = model.to(torch.device('cpu'))
        pred_s, pseudo_ys = model.classifier_prediction(x_input)
        pred_t, pseudo_yt = model.ensemble_prediction(x_input, y_input, torch.tensor(x_target).float())
        # select the data with high confidence
        conf_index_s = get_index_each_label(num_labels, top_s, pred_s, pseudo_ys)
        conf_index_t = get_index_each_label(num_labels, top_t, pred_t, pseudo_yt)
        if m == 1:
            x_inter = np.vstack([x_source[conf_index_s], x_target[conf_index_t]])
            y_inter = np.hstack([y_source[conf_index_s], pseudo_yt[conf_index_t]])
        else:
            x_inter = np.vstack([x_inter[conf_index_s], x_target[conf_index_t]])
            y_inter = np.hstack([y_inter[conf_index_s], pseudo_yt[conf_index_t]])
        print(f'top_s {top_s}, top_t {top_t}, x_inter size {x_inter.shape[0]}')
        model, _ = train_classifier(model, x_inter, y_inter, n_epochs, weight_decay)
        all_model.append(model)
    return all_model, None


def GIFT(x_all, y_all, iters, adapt_lmbda=3, hidden_dim=32, n_epochs=100, weight_decay=1e-3) -> (list, None):
    """
    Abnar proposed
    Gradual Domain Adaptation in the Wild:When Intermediate Distributions are Absent
    https://arxiv.org/abs/2106.06080
    @memo
    two-moon dataset example needs StandardScaler to each domain and 1 hidden layer, 32 nodes.
    @param
    iters: int, how many times lambda update
    adapt_lmbda: int, how many times update student model for synthesis data
    """
    # GIFT does not need intermediate dataset
    x_source, y_source = x_all[0].copy(), y_all[0].copy()
    x_target = x_all[-1].copy()
    input_dim = _check_input_dim(x_all[0])
    model = MLP(num_labels=np.unique(y_source).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)

    def align(ys, yt):
        index_s = np.arange(ys.shape[0])
        index_t = []
        for i in index_s:
            indices = np.arange(yt.size)
            indices = np.random.permutation(indices)
            index = np.argmax(ys[i] == yt[indices])
            index_t.append(indices[index])
        index_t = np.array(index_t)
        return index_s, index_t

    teacher_model, _ = train_classifier(model, x_source, y_source, n_epochs, weight_decay)
    all_model = [teacher_model]

    for i in tqdm(range(1, iters+1)):
        lmbda = (1.0 / iters) * i
        student_model = deepcopy(teacher_model)
        for j in range(adapt_lmbda):
            with torch.no_grad():
                zs = student_model.fc(util.torch_to(torch.tensor(x_source).float()))
                zt = teacher_model.fc(util.torch_to(torch.tensor(x_target).float()))
                pred_yt = teacher_model.pred(zt)
                pred_yt = torch.Tensor.cpu(pred_yt.argmax(dim=1)).numpy()
            index_s, index_t = align(y_source, pred_yt)
            zi = torch.vstack([(1.0 - lmbda) * zs[i] + lmbda * zt[j] for i,j in zip(index_s, index_t)])
            # update student model with pseudo label
            pseudo_y, conf_index = get_pseudo_y(teacher_model, zi, GIFT=True)
            student_model, _ = train_classifier(student_model, zi[conf_index], pseudo_y[conf_index], n_epochs, weight_decay, GIFT=True)
        teacher_model = deepcopy(student_model)
        all_model.append(teacher_model)
    return all_model, None
