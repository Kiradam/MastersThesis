import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data


def reader(path: str) -> dict:
    """
    Reading function

    :param path: relative/raw path to folder containing relevant csv-s
    :return: dict with loaded data
    """

    result = dict()
    df_features = pd.read_csv(path+"elliptic_txs_features.csv", header=None)
    df_edges = pd.read_csv(path+"elliptic_txs_edgelist.csv")
    df_classes = pd.read_csv(path+"elliptic_txs_classes.csv")
    result['features'] = df_features
    result['edges'] = df_edges
    result['classes'] = df_classes

    return result


def data_prep(featuresdf: pd.DataFrame,
              edgesdf: pd.DataFrame,
              classesdf: pd.DataFrame,
              time_seg: int,
              directed: bool = True,
              test_size: float = 0.15) -> tuple:

    """
    Prepares data for analysis with a given time segment

    :param featuresdf: df containing transaction features
    :param edgesdf: df containing edges
    :param classesdf: df containing classes
    :param time_seg: int for choosing time segment
    :param directed: bool for directed graph
    :param test_size: test size percentage
    :return: tuple with all data
    """

    # loading data
    df_features = featuresdf.copy()
    df_features = df_features.loc[df_features[1] == time_seg]
    time_id = df_features[0]
    df_edges = edgesdf.copy()
    df_edges = df_edges.loc[df_edges['txId1'].isin(time_id) | df_edges['txId2'].isin(time_id)]
    df_classes = classesdf.copy()
    df_classes = df_classes.loc[df_classes['txId'].isin(time_id)]
    df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0})

    # merging data
    df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
    df_merge = df_merge.sort_values(0).reset_index(drop=True)

    # storing classified unclassified nodes separately for training and testing purposes
    classified = df_merge.loc[df_merge['class'].loc[df_merge['class'] != 2].index].drop('txId', axis=1)
    unclassified = df_merge.loc[df_merge['class'].loc[df_merge['class'] == 2].index].drop('txId', axis=1)
    classified_edges = df_edges.loc[df_edges['txId1'].isin(classified[0]) & df_edges['txId2'].isin(classified[0])]
    unclassifed_edges = df_edges.loc[df_edges['txId1'].isin(unclassified[0]) | df_edges['txId2'].isin(unclassified[0])]

    # free memory
    del df_features, df_classes

    # dict for edge indexing based on nodes
    nodes = df_merge[0].values
    map_id = {j: i for i, j in enumerate(nodes)}  # mapping nodes to indexes

    # mapping transaction id-s based on nodes
    edges = df_edges.copy()
    edges.txId1 = edges.txId1.map(map_id)
    edges.txId2 = edges.txId2.map(map_id)
    edges = edges.astype(int)

    edge_index = np.array(edges.values).T

    # undirected graph option
    if not directed:
        edge_index_ = np.array([edge_index[1, :], edge_index[0, :]])
        edge_index = np.concatenate((edge_index, edge_index_), axis=1)

    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

    # mapping txIds to corresponding indexes, to pass node features to the model
    node_features = df_merge.drop(['txId'], axis=1).copy()
    node_features[0] = node_features[0].map(map_id)
    classified_idx = node_features['class'].loc[node_features['class'] != 2].index
    unclassified_idx = node_features['class'].loc[node_features['class'] == 2].index

    # replace unkown class with 0, to avoid having 3 classes, this data/labels never used in training
    node_features['class'] = node_features['class'].replace(2, 0)

    labels = node_features['class'].values
    node_features = torch.tensor(np.array(node_features.drop([0, 'class', 1], axis=1).values, dtype=np.double),
                                 dtype=torch.double)

    # converting data to PyGeometric graph data format
    data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                      y=torch.tensor(labels, dtype=torch.double))  # , adj= torch.from_numpy(np.array(adj))

    y_train = labels[classified_idx]

    # spliting train set and validation set
    X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(node_features[classified_idx], y_train,
                                                                                classified_idx, test_size=test_size,
                                                                                random_state=42, stratify=y_train)

    return data_train, X_train, X_valid, y_train, y_valid, train_idx, valid_idx, classified_idx, unclassified_idx

