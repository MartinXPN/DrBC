import copy
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense, LeakyReLU, LSTM, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.callbacks import CallbackList
from tqdm import tqdm

from drbcython import metrics, utils, graph, PrepareBatchGraph
from drbcpp.layers import DrBCRNN


# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+

aggregatorID = 2  # how to aggregate node neighbors, 0:sum; 1:mean; 2:GCN(weighted sum)


def pairwise_ranking_crossentropy_loss(y_true, y_pred):
    pred_betweenness = y_pred
    target_betweenness = tf.slice(y_true, begin=(0, 0), size=(-1, 1))
    src_ids = tf.cast(tf.reshape(tf.slice(y_true, begin=(0, 1), size=(-1, 5)), (-1,)), 'int32')
    tgt_ids = tf.cast(tf.reshape(tf.slice(y_true, begin=(0, 6), size=(-1, 5)), (-1,)), 'int32')

    labels = tf.nn.embedding_lookup(target_betweenness, src_ids) - tf.nn.embedding_lookup(target_betweenness, tgt_ids)
    preds = tf.nn.embedding_lookup(pred_betweenness, src_ids) - tf.nn.embedding_lookup(pred_betweenness, tgt_ids)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=tf.sigmoid(labels))


def create_drbc_model(node_feature_dim=3, aux_feature_dim=4, rnn_repetitions=5,
                      aggregation: str = 'max', combine='gru'):
    """
    :param node_feature_dim: initial node features, [Dc,1,1]
    :param aux_feature_dim: extra node features in the hidden layer in the decoder, [Dc,CI1,CI2,1]
    :param rnn_repetitions: how many loops are there in DrBCRNN
    :param aggregation: how to aggregate sequences after DrBCRNN {min, max, sum, mean, lstm}
    :param combine: how to combine in each iteration in DrBCRNN {structure2vec, graphsage, gru}
    :return: DrBC tf.keras model
    """
    input_node_features = Input(shape=(node_feature_dim,), name='node_features')
    input_aux_features = Input(shape=(aux_feature_dim,), name='aux_features')
    input_n2n = Input(shape=(None,), sparse=True, name='n2n_sum')

    node_features = Dense(units=128)(input_node_features)
    node_features = LeakyReLU()(node_features)
    node_features = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalize_node_features')(node_features)

    n2n_features = DrBCRNN(units=128, repetitions=rnn_repetitions, combine=combine, return_sequences=aggregation is not None)([input_n2n, node_features])
    if aggregation == 'max':        n2n_features = Lambda(lambda x: tf.reduce_max(x, axis=-1), name='aggregate')(n2n_features)
    elif aggregation == 'min':      n2n_features = Lambda(lambda x: tf.reduce_min(x, axis=-1), name='aggregate')(n2n_features)
    elif aggregation == 'sum':      n2n_features = Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='aggregate')(n2n_features)
    elif aggregation == 'mean':     n2n_features = Lambda(lambda x: tf.reduce_mean(x, axis=-1), name='aggregate')(n2n_features)
    elif aggregation == 'lstm':
        n2n_features = LSTM(units=128, return_sequences=True)(n2n_features)
        n2n_features = Attention()([n2n_features, n2n_features, n2n_features])
        n2n_features = Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='aggregate')(n2n_features)
    n2n_features = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalize_n2n')(n2n_features)

    all_features = Concatenate(axis=-1)([n2n_features, input_aux_features])
    top = Dense(64)(all_features)
    top = LeakyReLU()(top)
    out = Dense(1)(top)

    return Model(inputs=[input_node_features, input_aux_features, input_n2n], outputs=out, name='DrBC')


class DataGenerator(Sequence):
    def __init__(self, tag: str = 'generator', graph_type: str = '',
                 min_nodes: int = 0, max_nodes: int = 0,
                 nb_graphs: int = 1, graphs_per_batch: int = 1, nb_batches: int = 1,
                 include_idx_map: bool = False, random_samples: bool = True,
                 log_betweenness: bool = True, compute_betweenness: bool = True):
        self.utils = utils.py_Utils()
        self.graphs = graph.py_GSet()
        self.count: int = 0
        self.betweenness: List[float] = []
        self.tag: str = tag
        self.graph_type: str = graph_type
        self.min_nodes: int = min_nodes
        self.max_nodes: int = max_nodes
        self.nb_graphs: int = nb_graphs
        self.graphs_per_batch: int = graphs_per_batch
        self.nb_batches: int = nb_batches
        self.include_idx_map: bool = include_idx_map
        self.random_samples: bool = random_samples
        self.log_betweenness: bool = log_betweenness
        self.compute_betweenness = compute_betweenness

    def __len__(self) -> int:
        return self.nb_batches

    def get_batch(self, graphs, ids: List[int]):
        label = []
        for i in ids:
            label += self.betweenness[i]
        label = np.array(label)

        batch_graph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        batch_graph.SetupBatchGraph(graphs)
        assert (len(batch_graph.pair_ids_src) == len(batch_graph.pair_ids_tgt))

        batch_size = len(label)
        x = [
            np.array(batch_graph.node_feat),
            np.array(batch_graph.aux_feat),
            batch_graph.n2nsum_param
        ]
        y = np.concatenate([
            np.reshape(label, (batch_size, 1)),
            np.reshape(batch_graph.pair_ids_src, (batch_size, -1)),
            np.reshape(batch_graph.pair_ids_tgt, (batch_size, -1)),
        ], axis=-1)
        return (x, y, batch_graph.idx_map_list[0]) if self.include_idx_map else (x, y)

    def __getitem__(self, index: int):
        if self.random_samples:
            g_list, id_list = self.graphs.Sample_Batch(self.graphs_per_batch)
            return self.get_batch(graphs=g_list, ids=id_list)
        return self.get_batch(graphs=[self.graphs.Get(index)], ids=[index])

    @staticmethod
    def gen_network(g):  # networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            a = np.array(a)
            b = np.array(b)
        else:
            a = np.array([0])
            b = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), a, b)

    def gen_graph(self):
        cur_n = np.random.randint(self.min_nodes, self.max_nodes)
        if self.graph_type == 'erdos_renyi':        return nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.graph_type == 'small-world':      return nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.graph_type == 'barabasi_albert':  return nx.barabasi_albert_graph(n=cur_n, m=4)
        elif self.graph_type == 'powerlaw':         return nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        raise ValueError(f'{self.graph_type} graph type is not supported yet')

    def add_graph(self, g):
        t = self.count
        self.count += 1
        net = self.gen_network(g)
        self.graphs.InsertGraph(t, net)

        if self.compute_betweenness:
            bc = self.utils.Betweenness(net)
            bc_log = self.utils.bc_log
            self.betweenness.append(bc_log if self.log_betweenness else bc)

    def gen_new_graphs(self):
        self.clear()
        for _ in tqdm(range(self.nb_graphs), desc=f'{self.tag}: generating new graphs...'):
            g = self.gen_graph()
            self.add_graph(g)

    def clear(self):
        self.count = 0
        self.graphs.Clear()
        self.betweenness = []


class EvaluateCallback(Callback):
    def __init__(self, data_generator, prepend_str: str = 'val_'):
        super().__init__()
        self.data_generator = data_generator
        self.prepend_str = prepend_str
        self.metrics = metrics.py_Metrics()
        self._supports_tf_logs = True

    def evaluate(self):
        epoch_logs = {
            f'{self.prepend_str}top0.01': [],
            f'{self.prepend_str}top0.05': [],
            f'{self.prepend_str}top0.1': [],
            f'{self.prepend_str}kendal': [],
        }
        for gid, (x, y, idx_map) in enumerate(self.data_generator):
            result = self.model.predict_on_batch(x=x).flatten()
            betw_predict = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                            for i, pred_betweenness in enumerate(result)]

            betw_label = self.data_generator.betweenness[gid]
            epoch_logs[f'{self.prepend_str}top0.01'].append(self.metrics.RankTopK(betw_label, betw_predict, 0.01))
            epoch_logs[f'{self.prepend_str}top0.05'].append(self.metrics.RankTopK(betw_label, betw_predict, 0.05))
            epoch_logs[f'{self.prepend_str}top0.1'].append(self.metrics.RankTopK(betw_label, betw_predict, 0.1))
            epoch_logs[f'{self.prepend_str}kendal'].append(self.metrics.RankKendal(betw_label, betw_predict))
        return {k: np.mean(val) for k, val in epoch_logs.items()}

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        logs = logs or {}
        logs.update(self.evaluate())


class BetLearn:
    def __init__(self, min_nodes, max_nodes, nb_train_graphs, nb_valid_graphs, graphs_per_batch, nb_batches,
                 graph_type='powerlaw', optimizer='adam', aggregation='lstm', combine='gru'):
        """
        :param min_nodes: minimum training scale (node set size)
        :param max_nodes: maximum training scale (node set size)
        :param nb_train_graphs: number of train graphs
        :param nb_valid_graphs: number of validation graphs
        :param graphs_per_batch: number of graphs sampled per batch
        :param nb_batches: number of batches to process per each training epoch
        :param graph_type: {powerlaw, erdos_renyi, powerlaw, small-world, barabasi_albert}
        :param optimizer: any tf.keras supported optimizer
        :param aggregation: how to aggregate sequences after DrBCRNN {min, max, sum, mean, lstm}
        :param combine: how to combine in each iteration in DrBCRNN {structure2vec, graphsage, gru}
        """
        self.experiment_path = Path('../experiments') / datetime.now().replace(microsecond=0).isoformat()
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_generator = DataGenerator(tag='Train', graph_type=graph_type, min_nodes=min_nodes, max_nodes=max_nodes, nb_graphs=nb_train_graphs, graphs_per_batch=graphs_per_batch, nb_batches=nb_batches, include_idx_map=False, random_samples=True, log_betweenness=True)
        self.valid_generator = DataGenerator(tag='Valid', graph_type=graph_type, min_nodes=min_nodes, max_nodes=max_nodes, nb_graphs=nb_valid_graphs, graphs_per_batch=1, nb_batches=nb_valid_graphs, include_idx_map=True, random_samples=False, log_betweenness=False)

        self.model = create_drbc_model(aggregation=aggregation, combine=combine)
        self.model.compile(optimizer=optimizer, loss=pairwise_ranking_crossentropy_loss)
        self.model.summary()
        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')

    def predict(self, gid):
        x, y, idx_map = self.valid_generator[gid]
        result = self.model.predict_on_batch(x=x).flatten()

        # idx_map[i] >= 0:  # corresponds to nodes with 0.0 betw_log value
        result_output = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                         for i, pred_betweenness in enumerate(result)]
        return result_output

    def train(self, epochs):
        """
        functional API with model.fit doesn't support sparse tensors with the current implementation =>
        we write the training loop ourselves
        """
        callbacks = CallbackList([
            EvaluateCallback(self.valid_generator, prepend_str='val_'),
            TensorBoard(self.log_dir, profile_batch=0),
            ModelCheckpoint(self.model_save_path / 'best.h5py', monitor='val_kendal', save_best_only=True, verbose=1, mode='max'),
            EarlyStopping(monitor='val_kendal', patience=5, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_kendal', patience=2, factor=0.5, mode='max'),
        ],  add_history=True, add_progbar=True, verbose=1,
            model=self.model,
            epochs=epochs, steps=len(self.train_generator))

        callbacks.on_train_begin()
        for epoch in range(epochs):
            if epoch % 5 == 0:
                self.train_generator.gen_new_graphs()
                self.valid_generator.gen_new_graphs()

            callbacks.on_epoch_begin(epoch)
            [c.on_train_begin() for c in callbacks]
            for batch, (x, y) in enumerate(self.train_generator):
                callbacks.on_train_batch_begin(batch)
                logs = self.model.train_on_batch(x, y, return_dict=True)
                callbacks.on_train_batch_end(batch, logs)

            epoch_logs = copy.copy(logs)
            callbacks.on_epoch_end(epoch, logs=epoch_logs)
            pd.DataFrame(self.model.history.history).to_csv(self.log_dir / 'history.csv', index=False)
            if self.model.stop_training:
                break

        callbacks.on_train_end(copy.copy(epoch_logs))
        print(self.model.history.history)
