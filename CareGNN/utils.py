import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
import copy as cp

from pandas import DataFrame
from pandas._typing import NDFrameT
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from collections import defaultdict
import pandas as pd
from typing import Tuple, Any

"""
	Utility functions to handle data and evaluate model.
"""


def load_data(prefix: str = '../data/', edges_path: str = 'hdbscan_70.csv') -> Tuple[
	Any, Any, DataFrame | NDFrameT | None, Any]:
	"""
	Data loader for CARE/SAGE-GNN

	:param prefix: path to feature, label and edge data
	:param edges_path: name of edge color file
	:return:
	"""

	classes = pd.read_csv(prefix + 'elliptic_txs_classes.csv')
	classes = classes.loc[classes['class'] != 'unknown']
	edges = pd.read_csv(prefix + 'elliptic_txs_edgelist.csv')
	edges = edges.merge(classes, left_on='txId1', right_on='txId', how='inner')
	edges = edges.merge(classes, left_on='txId2', right_on='txId', how='inner')
	edges.drop(['txId_x', 'class_x', 'txId_y', 'class_y'], axis=1, inplace=True)

	features = pd.read_csv(prefix + 'elliptic_txs_features.csv', header=None)
	classes = pd.read_csv(prefix + 'elliptic_txs_classes.csv')
	data = features.merge(classes, left_on=0, right_on='txId', how='left')
	data = data.loc[data['class'] != 'unknown']
	all_nodes = data['txId']
	rename = dict()
	for new, old in enumerate(data['txId']):
		rename[old] = new

	data.drop([1, 'txId', 0], axis=1, inplace=True)

	edges['txId1'] = edges['txId1'].apply(lambda x: rename[x])
	edges['txId2'] = edges['txId2'].apply(lambda x: rename[x])
	all_edges = pd.read_csv(prefix + edges_path).drop(['Unnamed: 0'], axis=1)

	all_nodes = all_nodes.apply(lambda x: rename[x])
	all_edges["txId1"] = all_edges["txId1"].apply(lambda x: rename[x])
	all_edges["txId2"] = all_edges["txId2"].apply(lambda x: rename[x])
	labels = ((data['class'].astype(int) - 2) * (-1)).to_numpy()
	feat_data = data.drop('class', axis=1).to_numpy()
	return all_nodes, feat_data, all_edges, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


def pos_neg_split(nodes, labels):
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
	pos_nodes = []
	neg_nodes = cp.deepcopy(nodes)
	aux_nodes = cp.deepcopy(nodes)
	for idx, label in enumerate(labels):
		if label == 1:
			pos_nodes.append(aux_nodes[idx])
			neg_nodes.remove(aux_nodes[idx])

	return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
	"""
	Under-sample the negative nodes
	:param pos_nodes: a list of positive nodes
	:param neg_nodes: a list negative nodes
	:param scale: the under-sampling scale
	:return: a list of under-sampled batch nodes
	"""

	aux_nodes = cp.deepcopy(neg_nodes)
	aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes)*scale))
	batch_nodes = pos_nodes + aux_nodes

	return batch_nodes


def test_sage(test_cases, labels, model, batch_size):
	"""
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	f1_label1 = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	gnn_list = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)
		gnn_label = gnn_prob.data.cpu().numpy().argmax(axis=1)
		f1_gnn += f1_score(batch_label, gnn_label, average="macro")
		label1_gnn = []
		for i, j in enumerate(batch_label):
			if j == 1:
				label1_gnn.append(gnn_label[i])
		f1_label1 += f1_score(len(label1_gnn) * [1], label1_gnn, average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

	label1_gnn = []
	for i, j in enumerate(labels):
		if j == 1:
			label1_gnn.append(gnn_list[i])

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	print(f"GNN F1: {f1_gnn / test_batch_num:.4f}")
	print(f"GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
	print(f"GNN Recall: {recall_gnn / test_batch_num:.4f}")
	print(f"GNN auc: {auc_gnn:.4f}")
	print(f"GNN ap: {ap_gnn:.4f}")

	return auc_gnn, f1_gnn/test_batch_num, ap_gnn


def test_care(test_cases, labels, model, batch_size):
	"""
	Test the performance of CARE-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	f1_label1 = 0.0
	acc_label1 = 0.00
	recall_label1 = 0.0
	gnn_list = []
	label_list1 = []

	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

		f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
		recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
		label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))

	print(f"GNN F1: {f1_gnn / test_batch_num:.4f}")
	print(f"GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
	print(f"GNN Recall: {recall_gnn / test_batch_num:.4f}")
	print(f"GNN auc: {auc_gnn:.4f}")
	print(f"GNN ap: {ap_gnn:.4f}")

	return auc_gnn, f1_gnn / test_batch_num, ap_gnn
