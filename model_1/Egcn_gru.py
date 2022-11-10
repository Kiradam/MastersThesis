from memory_profiler import profile
import torch
import torch_geometric
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import gc
from model_1.net import Net
from typing import List
import numpy as np
from torch_geometric.nn import GCNConv


@profile
def egcn_train(networks: List[Net],
               data_train: List[torch_geometric.data.Data],
               layers_num: int,
               train_idx: List[np.ndarray],
               conv1_size: int = 165, conv2_size: int = 128,
               epnum: int = 2000,
               print_res: bool = True,
               lr: float = 0.0001,
               weight_decay: float = 1e-5):
    ret = list()
    weights_1 = list()
    weights_2 = list()
    bias_1 = list()
    bias_2 = list()
    size1 = conv1_size*conv2_size
    for i in networks:
        for name, value in i.named_parameters():
            if name == 'conv1.bias':
                bias_1.append(value)
            elif name == 'conv1.lin.weight':
                weights_1.append(value)
            elif name == 'conv2.bias':
                bias_2.append(value)
            else:
                weights_2.append(value)
    weights_1_input = list()
    weights_2_input = list()
    for i in weights_1:
        weights_1_input.append(i.detach().float().numpy().reshape(1, size1))
    for i in weights_2:
        weights_2_input.append(i.detach().numpy().reshape(1, conv2_size))
    weights_1_network = torch.nn.GRU(size1, size1, layers_num)
    weights_2_network = torch.nn.GRU(conv2_size, conv2_size, layers_num)
    nets = [weights_1_network, weights_2_network]
    for i in range(1):
        gc.collect()
        networks_copy = networks[:]
        # allocating Tensor to gpu/cpu, casting to double
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(gc.get_stats())

        model = nets[i]
        model.double()

        # configuring model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # in article lr= 0.0001
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = torch.nn.BCELoss()
        # setting model to training mode
        model.train()

        # optimalization
        for epoch in range(epnum):  # in article epochs go to 2000
            optimizer.zero_grad()  # set optimized tensors' grad to 0

            if i == 0:
                out = model(torch.tensor(np.array(weights_1_input)))
                gcn = GCNConv(165, 128)
                loss = 0
                for k in range(len(networks_copy)):
                    gcn.lin.weight = torch.nn.Parameter(torch.reshape(out[0][k][0], (128, 165)))
                    timestamp_data = data_train[k]
                    timestamp_data.to(device)
                    networks_copy[k].conv1 = gcn
                    out2 = networks_copy[k](timestamp_data)
                    out2 = out2.reshape((timestamp_data.x.shape[0]))
                    if loss == 0:
                        loss = criterion(out2[train_idx[k]], timestamp_data.y[train_idx[k]])
                    else:
                        loss = loss + criterion(out2[train_idx[k]], timestamp_data.y[train_idx[k]])
                auc = roc_auc_score(timestamp_data.y.detach().cpu().numpy()[train_idx],
                                    out2.detach().cpu().numpy()[train_idx])  # [train_idx] [train_idx[k]]
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0 and print_res:
                    print("epoch: {} - loss: {} - roc: {}".format(epoch, loss.item(), auc))
            else:
                out = model(torch.tensor(np.array(weights_2_input)))
                gcn = GCNConv(128, 1)
                loss = 0
                for k in range(len(networks_copy)):
                    gcn.lin.weight = torch.nn.Parameter(torch.reshape(out[0][k][0], (1, 128)))
                    timestamp_data = data_train[k]
                    timestamp_data.to(device)
                    networks_copy[k].conv2 = gcn
                    out2 = networks_copy[k](timestamp_data)
                    out2 = out2.reshape((timestamp_data.x.shape[0]))
                    if loss == 0:
                        loss = criterion(out2[train_idx[k]], timestamp_data.y[train_idx[k]])
                    else:
                        loss = loss+criterion(out2[train_idx[k]], timestamp_data.y[train_idx[k]])
                auc = roc_auc_score(timestamp_data.y.detach().cpu().numpy()[train_idx],
                                    out2.detach().cpu().numpy()[train_idx])  # [train_idx] [train_idx[k]]
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0 and print_res:
                    print("epoch: {} - loss: {} - roc: {}".format(epoch, loss.item(), auc))
        ret.append(model)
    return ret

