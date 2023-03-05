from GCN.net import Net
from GCN.data_prep import data_prep
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, average_precision_score
import torch
import torch_geometric
import time


def train_1segment(data_train: torch_geometric.data.Data,
                   train_idx: np.ndarray,
                   valid_idx: np.ndarray,
                   epnum_gcn: int = 70,
                   print_res: bool = True,
                   *args,
                   **kwargs) -> tuple:
    """
    Training the model for 1 time-segment (needs to be prefiltered)

    :param data_train: PyGeometric graph containing all data
    :param train_idx: ndarray containing node ID-s for train
    :param valid_idx: ndarray containing node ID-s for validation
    :param epnum_gcn: number of epochs
    :param print_res: bool whether to print intermediate results
    :return: tuple with confusion matrices and trained model
    """

    # allocating Tensor to gpu/cpu, casting to double
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.double()
    data_train = data_train.to(device)

    # configuring model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # in article lr= 0.0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.BCELoss()

    # setting model to training mode
    model.train()

    # optimalization
    for epoch in range(epnum_gcn):  # in article epochs go to 2000
        optimizer.zero_grad()  # set optimized tensors' grad to 0
        out = model(data_train)
        out = out.reshape((data_train.x.shape[0]))
        loss = criterion(out[train_idx], data_train.y[train_idx])
        auc = roc_auc_score(data_train.y.detach().cpu().numpy()[train_idx],
                            out.detach().cpu().numpy()[train_idx])  # [train_idx]
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 and print_res:
            print("epoch gcn: {} - loss: {} - roc: {}".format(epoch, loss.item(), auc))

    # labeling results
    preds = model(data_train)
    preds = preds.detach().cpu().numpy()

    out_labels = preds > 0.6

    # accuracy calculations
    train_acc = accuracy_score(data_train.y.detach().cpu().numpy()[train_idx], out_labels[train_idx])
    train_auc = roc_auc_score(data_train.y.detach().cpu().numpy()[train_idx], preds[train_idx])
    train_f1 = f1_score(data_train.y.detach().cpu().numpy()[train_idx], out_labels[train_idx])
    train_ap = average_precision_score(data_train.y.detach().cpu().numpy()[train_idx], out_labels[train_idx])
    if print_res:
        print("train accuracy: ", train_acc)
        print("train AUC: ", train_auc)
        print("train F1: ", train_f1)
        print("train ap: ", train_ap)
    valid_auc = roc_auc_score(data_train.y.detach().cpu().numpy()[valid_idx], preds[valid_idx])
    out_labels = out.detach().cpu().numpy() > 0.6
    valid_acc = accuracy_score(data_train.y.detach().cpu().numpy()[valid_idx], out_labels[valid_idx])
    valid_f1 = f1_score(data_train.y.detach().cpu().numpy()[valid_idx], out_labels[valid_idx])
    valid_ap = average_precision_score(data_train.y.detach().cpu().numpy()[valid_idx], out_labels[valid_idx])
    if print_res:
        print("valid accuracy: ", valid_acc)
        print("valid AUC: ", valid_auc)
        print("valid F1: ", valid_f1)
        print("valid ap: ", valid_ap)

    # confusion matrices
    cm_valid = confusion_matrix(data_train.y.detach().cpu().numpy()[valid_idx], out_labels[valid_idx])
    cm_train = confusion_matrix(data_train.y.detach().cpu().numpy()[train_idx], out_labels[train_idx])

    return model, cm_train, cm_valid, train_f1, valid_f1


def multiproc(d: dict, seg: int = 35, **kwargs) -> tuple:
    """
    Helper function for multiprocessing

    :param d: dict with loaded data
    :param seg: time segment
    :return: tuple with relevant results
    """
    data_train, X_train, X_valid, y_train, y_valid, train_idx, valid_idx, classified_idx, unclassified_idx = data_prep(
        d['features'], d['edges'], d['classes'], seg, **kwargs)

    return train_1segment(data_train, train_idx, valid_idx, **kwargs)


def timer(func):
    """
    Timer decorator
    :param func: function to time
    :return: wrapper
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'\n{func.__name__} took {round(end-start, 2)} seconds to finish.')
        return res
    return wrapper
