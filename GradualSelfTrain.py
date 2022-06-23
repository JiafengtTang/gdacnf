import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# my module
from util import torch_to, preprocess_input, rounded_statistics


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


# +
def get_pseudo_y(model: nn.Module, x: torch.Tensor, confidence_q: float=0.1, GIFT: bool=False) -> (np.ndarray, np.ndarray):
    """ remove less confidence sample """
    dataset = preprocess_input(x)
    model, _x = torch_to(model, dataset.tensors[0])
    with torch.no_grad():
        logits = model(_x) if not GIFT else model.pred(_x)
        logits = nn.functional.softmax(logits, dim=1)
        confidence = np.array(torch.Tensor.cpu(logits.amax(dim=1) - logits.amin(dim=1)))
        alpha = np.quantile(confidence, confidence_q)
        conf_index = np.argwhere(confidence >= alpha)[:,0]
        pseudo_y = logits.argmax(dim=1)
    return pseudo_y.detach().cpu().numpy(), conf_index


def calc_accuracy(model, x, y):
    dataset = preprocess_input(x)
    with torch.no_grad():
        model, _x = torch_to(model, dataset.tensors[0])
        pred = model(_x)
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.array(torch.Tensor.cpu(pred.argmax(dim=1)))
    return accuracy_score(y, pred.squeeze())


def train_classifier(clf, x, y, n_epochs=100, weight_decay=1e-3, GIFT=False):
    model = deepcopy(clf)
    model = torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_f = nn.CrossEntropyLoss()
    batch_size = 512
    dataset = preprocess_input(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = []
    for e in range(n_epochs):
        running_loss = 0
        for x_sample, y_sample in train_loader:
            x_sample, y_sample = torch_to(x_sample, y_sample)
            optimizer.zero_grad()
            y_pred = model(x_sample) if not GIFT else model.pred(x_sample)
            loss = loss_f(y_pred, y_sample)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / batch_size
        loss_history.append(running_loss)
    return model, loss_history


def compare_model_parameter(model1, model2):
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    delta_norm = 0
    for p1, p2 in zip(params1, params2):
        p1 = p1.view(p1.shape[0], -1)
        p2 = p2.view(p2.shape[0], -1)
        norm = torch.linalg.norm(p1-p2, ord='fro')
        norm = norm.detach().cpu().item()
        delta_norm += norm
    return delta_norm


def gradual_self_train(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    @retrun
    student_model: nn.Module, target model \theta^{(K)}
    reults: dict, model and items used for training
    """
    results = {}
    if np.ndim(x_all[-1]) == 4:
        input_dim = x_all[-1].shape[-2:]
    else:
        input_dim = x_all[-1].shape[1]
    model = MLP(num_labels=np.unique(y_all[-1]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    student_model = deepcopy(model)
    teacher_model = deepcopy(model)
    student_model, loss_history = train_classifier(student_model, x_all[0], y_all[0], n_epochs, weight_decay)
    results[0] = [student_model, x_all[0], y_all[0], np.arange(len(y_all[0]))]
    #delta_norm = []
    for j, x in enumerate(tqdm(x_all[1:])):
        teacher_model.load_state_dict(student_model.state_dict())
        pseudo_y, conf_index = get_pseudo_y(teacher_model, x)
        student_model, loss_history = train_classifier(student_model, x[conf_index], pseudo_y[conf_index], n_epochs, weight_decay)
        results[j+1] = [student_model, x, pseudo_y, conf_index]
        #delta_norm.append(compare_model_parameter(teacher_model, student_model))
    return student_model, results#, np.mean(delta_norm)


def cycle_consistency_self_train(x_all, y_all, x_eval, y_eval, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    # train forward
    clf_forward, res_forward = gradual_self_train(x_all, y_all, hidden_dim, n_epochs, weight_decay)
    # train backward with pseudo label
    x_all_inv, y_all_inv = deepcopy(x_all[::-1]), deepcopy(y_all[::-1])
    pseudo_y, conf_index = get_pseudo_y(clf_forward, x_all_inv[0])
    x_all_inv[0] = x_all_inv[0][conf_index]
    y_all_inv[0] = pseudo_y[conf_index]
    clf_backward, res_backward = gradual_self_train(x_all_inv, y_all_inv, hidden_dim, n_epochs, weight_decay)
    # check consistency
    f_acc = calc_accuracy(clf_forward, x_eval, y_eval)
    b_acc = calc_accuracy(clf_backward, x_all[0], y_all[0])
    return res_forward, res_backward, f_acc, b_acc


def eval_baseline(x_all, y_all, x_eval, y_eval, repeat=3, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """ evaluate source only and target onnly model """
    s_acc, t_acc = list(), list()
    if np.ndim(x_all[-1]) == 4:
        input_dim = x_all[-1].shape[-2:]
    else:
        input_dim = x_all[-1].shape[1]
    for r in range(repeat):
        clf = MLP(num_labels=np.unique(y_all[-1]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
        s_clf, _ = train_classifier(clf, x_all[0], y_all[0], n_epochs, weight_decay)
        s_acc.append(calc_accuracy(s_clf, x_eval, y_eval))
        t_clf, _ = train_classifier(clf, x_all[-1], y_all[-1], n_epochs, weight_decay)
        t_acc.append(calc_accuracy(t_clf, x_eval, y_eval))
    print('source model accuracy {}'.format(rounded_statistics(s_acc)))
    print('target model accuracy {}'.format(rounded_statistics(t_acc)))
    return s_acc, t_acc


# -

def GIFT(x_all, y_all, iters, adapt_lmbda, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    Abnar proposed
    Gradual Domain Adaptation in the Wild:When Intermediate Distributions are Absent
    https://arxiv.org/abs/2106.06080
    @memo
    moon toy data example needs StandardScaler to each domain
    @param
    iters: int, how many times lambda update
    adapt_lmbda: int, how many times update student model for synthesis data
    """
    # GIFT does not need intermediate data
    x_source, y_source = x_all[0].copy(), y_all[0].copy()
    x_target = x_all[-1].copy()
    model = MLP(num_labels=np.unique(y_source).shape[0], input_dim=x_source.shape[1], hidden_dim=hidden_dim)

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

    for i in tqdm(range(1, iters+1)):
        lmbda = (1.0 / iters) * i
        student_model = deepcopy(teacher_model)
        for j in range(adapt_lmbda):
            with torch.no_grad():
                zs = student_model.fc(torch_to(torch.tensor(x_source)))
                zt = teacher_model.fc(torch_to(torch.tensor(x_target)))
                pred_yt = teacher_model.pred(zt)
                pred_yt = torch.Tensor.cpu(pred_yt.argmax(dim=1)).numpy()
            index_s, index_t = align(y_source, pred_yt)
            zi = torch.vstack([(1.0 - lmbda) * zs[i] + lmbda * zt[j] for i,j in zip(index_s, index_t)])
            # update student model with pseudo label
            pseudo_y, conf_index = get_pseudo_y(teacher_model, zi, GIFT=True)
            student_model, _ = train_classifier(model, zi[conf_index], pseudo_y[conf_index], n_epochs, weight_decay, GIFT=True)
        teacher_model = deepcopy(student_model)
    return teacher_model
