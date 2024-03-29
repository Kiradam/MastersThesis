from sklearn.model_selection import train_test_split
from utils import *
from layers import *
from graphsage import *
from model import *
import time
import pickle
from typing import Tuple, Union, List, DefaultDict
from enum import Enum


class ModelType(Enum):
    CARE = "CARE"
    SAGE = "SAGE"


class CareInterface:
    """
    Model interface for handling CARE and SAGE-GNN related tasks
    """
    _model_type = None
    _inter_agg = None
    _num_relations = None
    _relation_list = None
    _features = None
    _labels = None
    _cuda = None
    _model = None
    epochs_result = None
    _batch_size = None

    def __init__(self, model_type: ModelType, inter_agg: str = 'GNN') -> None:
        """
        Constructor for model interface

        :param model_type: type of model
        :param inter_agg: inter relation aggregator
        """
        self._model_type = model_type
        self._inter_agg = inter_agg
        self._cuda = torch.cuda.is_available()

    def preprocess(self,
                   all_nodes: pd.Series,
                   features: np.array,
                   edges: pd.DataFrame,
                   labels: np.array,
                   result: bool = False) -> Union[None, Tuple[List[DefaultDict[Any, set]], np.array, np.array]]:
        """
        Preprocessing function for appropriate data format

        :param all_nodes: series of node ID-s
        :param features: array of features
        :param edges: dataframe of edges and relations
        :param labels: array of labels
        :param result: bool for returning the processed dataset
        :return: None or processed dataset
        """

        self._num_relations = len(edges[edges.columns[2]].unique())
        relation_types = edges[edges.columns[2]].unique()
        relation_list = []

        for i in range(self._num_relations + 1):
            # using adding_set instead of lambda due to pickle conflict
            relation_list.append(defaultdict(adding_set()))

        for _, j in all_nodes.items():
            relation_list[0][j].add(j)
        for i, j in edges.iterrows():
            relation_list[0][j['txId1']].add(j['txId2'])
            relation_list[0][j['txId2']].add(j['txId1'])

        for k, l in zip(relation_types, range(self._num_relations)):
            for _, j in all_nodes.items():
                relation_list[l + 1][j].add(j)
            for i, j in edges.loc[edges[edges.columns[2]] == -1].iterrows():
                relation_list[l + 1][j['txId1']].add(j['txId2'])
                relation_list[l + 1][j['txId2']].add(j['txId1'])

        if result:
            return relation_list, features, labels
        else:
            self._relation_list = relation_list
            self._features = features
            self._labels = labels

    def fit(self,
            data_list,
            batch_size: int = 1024,
            lr: float = 0.01,
            lambda_1: float = 2,
            lambda_2: float = 1e-3,
            embedding_size: int = 64,
            num_epochs: int = 256,
            test_epochs: int = 3,
            test_size: float = 0.6,
            step_size: float = 2e-2,
            seed: int = 72,
            preprocessed: bool = False,
            optimize_on_ap: bool = True) -> None:
        """
        Fitting method of the interface

        :param data_list:
        :param batch_size: batch size
        :param lr: learning rate
        :param lambda_1: lambda 1
        :param lambda_2: lambda 2
        :param embedding_size: size of embedding
        :param num_epochs: number of epochs during fitting
        :param test_epochs: frequency of testing
        :param test_size: size of test set
        :param step_size: step size
        :param seed: random seed
        :param preprocessed: bool for getting preprocessed data
        :param optimize_on_ap: bool for saving model when test average precision is better
        """

        max_ap = 0
        self._batch_size = batch_size
        # load graph, feature, and label
        if not preprocessed:
            all_nodes = data_list[0]
            features = data_list[1]
            edges = data_list[2]
            labels = data_list[3]
            self.preprocess(all_nodes, features, edges, labels)
        index = list(range(len(self._labels)))

        epoch_test = defaultdict(lambda: [])
        # train_test split
        idx_train, idx_test, y_train, y_test = train_test_split(index,
                                                                self._labels,
                                                                stratify=self._labels,
                                                                test_size=test_size,
                                                                random_state=seed,
                                                                shuffle=True)

        # split pos neg sets for under-sampling
        train_pos, train_neg = pos_neg_split(idx_train, y_train)
        # initialize model input
        features = nn.Embedding(self._features.shape[0], self._features.shape[1])
        feat_data = normalize(self._features)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        labels = self._labels

        # build one-layer models
        if self._model_type.value == 'SAGE':
            adj_lists = self._relation_list[0]
            agg1 = MeanAggregator(features, cuda=self._cuda)
            enc1 = Encoder(features, feat_data.shape[1], embedding_size, adj_lists, agg1, gcn=True, cuda=self._cuda)
            enc1.num_samples = 5
            gnn_model = GraphSage(2, enc1)
            inter1 = None

        else:
            adj_lists = self._relation_list[1:]
            intra_list = []
            for _ in range(self._num_relations):
                intra_list.append(IntraAgg(features, feat_data.shape[1], cuda=self._cuda))
            inter1 = InterAgg(features,
                              feat_data.shape[1],
                              embedding_size,
                              adj_lists,
                              intra_list,
                              inter=self._inter_agg,
                              step_size=step_size,
                              cuda=self._cuda)
            gnn_model = OneLayerCARE(2, inter1, lambda_1)

        if self._cuda:
            gnn_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=lr,
                                     weight_decay=lambda_2)

        for epoch in range(num_epochs):

            # randomly under-sampling negative nodes for each epoch
            sampled_idx_train = undersample(train_pos, train_neg, scale=1)
            rd.shuffle(sampled_idx_train)

            # send number of batches to model to let the RLModule know the training progress
            num_batches = int(len(sampled_idx_train) / batch_size) + 1
            if self._model_type.value == 'CARE':
                inter1.batch_num = num_batches

            loss = 0.0
            epoch_time = 0

            # mini-batch training
            for batch in range(num_batches):

                start_time = time.time()
                i_start = batch * batch_size
                i_end = min((batch + 1) * batch_size, len(sampled_idx_train))
                batch_nodes = sampled_idx_train[i_start:i_end]
                batch_label = labels[np.array(batch_nodes)]
                optimizer.zero_grad()
                if self._cuda:
                    loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
                else:
                    loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
                loss.backward()
                optimizer.step()
                end_time = time.time()
                epoch_time += end_time - start_time
                loss += loss.item()

            print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

            # testing the model for every $test_epoch$ epoch
            if epoch % test_epochs == 0:
                if self._model_type.value == 'SAGE':
                    for i in test_sage(idx_test, y_test, gnn_model, batch_size):
                        epoch_test[epoch].append(i)
                else:
                    for i in test_care(idx_test, y_test, gnn_model, batch_size):
                        epoch_test[epoch].append(i)
                current_ap = epoch_test[epoch][2]
                if optimize_on_ap and max_ap < current_ap:
                    max_ap = current_ap
                    self._model = gnn_model
        self.epochs_result = pd.DataFrame(epoch_test).T.set_axis(['gnn_auc', 'gnn_f1', 'gnn_ap'], axis=1)
        if not optimize_on_ap:
            self._model = gnn_model

    def predict_proba(self, features: np.array, edges: pd.DataFrame) -> List[float]:
        """
        Predict probabilities

        :param features: features to test
        :param edges: edges
        :return: list of probabilities
        """
        all_nodes = pd.Series(range(len(features)))

        rel, feat, lab = self.preprocess(all_nodes, features, edges, [0] * len(features), result=True)

        features = nn.Embedding(feat.shape[0], feat.shape[1])
        feat_data = normalize(feat)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

        if self._model_type.value == 'SAGE':
            self._model.enc.features = features
            self._model.enc.adj_lists = rel[0]
        else:
            self._model.inter1.features = features
            self._model.inter1.adj_lists = rel[1:]

        idx_test = all_nodes.to_list()

        gnn_model = self._model
        batch_size = self._batch_size
        test_batch_num = int(len(idx_test) / batch_size) + 1
        gnn_list = []

        for iteration in range(test_batch_num):
            i_start = iteration * batch_size
            i_end = min((iteration + 1) * batch_size, len(idx_test))
            batch_nodes = idx_test[i_start:i_end]
            if self._model_type.value == 'SAGE':
                self._model.enc.num_sample = None
                gnn_prob = gnn_model.to_prob(batch_nodes)
            else:
                gnn_prob, label_prob1 = gnn_model.to_prob(batch_nodes, labels=[0] * (i_end - i_start), train_flag=False)

            gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

        return gnn_list

    def save_model(self, filename: str) -> None:
        """
        Function for saving model

        :param filename: path to save
        """

        pickle.dump(self._model, open(filename + '.sav', 'wb'))

    def load_model(self, filename: str, batch_size: int = 1024) -> None:
        """
        Function for loading model

        :param filename: path to model
        :param batch_size: batch size used for training
        """

        self._batch_size = batch_size
        self._model = pickle.load(open(filename + '.sav', 'rb'))


def adding_set() -> type:
    """
    Function for returning set type (pickle conflict)
    """
    return set


if __name__ == '__main__':
    # loading data
    list_of_nodes, feature_data, edge_colours, classes = load_data()

    # training CARE-GNN with hdbscan edges
    analysis_hdbscan = CareInterface(model_type=ModelType.CARE)
    analysis_hdbscan.fit([list_of_nodes, feature_data, edge_colours, classes], num_epochs=1024, optimize_on_ap=True)
    analysis_hdbscan.save_model('../models_result/care_gnn')
    analysis_hdbscan.epochs_result.to_csv('../models_result/care.csv')

    # training SAGE-GNN with hdbscan edges
    analysis_hdbscan = CareInterface(model_type=ModelType.SAGE)
    analysis_hdbscan.fit([list_of_nodes, feature_data, edge_colours, classes], num_epochs=1024, optimize_on_ap=True)
    analysis_hdbscan.save_model('../models_result/sage_gnn')
    analysis_hdbscan.epochs_result.to_csv('../models_result/sage.csv')

    # loading data with xgb predicted edges
    list_of_nodes, feature_data, edge_colours, classes = load_data(edges_path='xgb_graph.csv')

    # training CARE-GNN with xgb edges
    analysis_xgb = CareInterface(model_type=ModelType.CARE)
    analysis_xgb.fit([list_of_nodes, feature_data, edge_colours, classes], num_epochs=1024, optimize_on_ap=True)
    analysis_xgb.save_model('../models_result/care_gnn_xgb')
    analysis_xgb.epochs_result.to_csv('../models_result/care_xgb.csv')

    # training SAGE-GNN with xgb edges
    analysis_xgb = CareInterface(model_type=ModelType.SAGE)
    analysis_xgb.fit([list_of_nodes, feature_data, edge_colours, classes], num_epochs=1024, optimize_on_ap=True)
    analysis_xgb.save_model('../models_result/sage_gnn_xgb')
    analysis_xgb.epochs_result.to_csv('../models_result/sage_xgb.csv')
