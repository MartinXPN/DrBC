import copy
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd
from aim import Session
from aim.tensorflow import AimCallback
from tensorflow.keras.callbacks import CallbackList, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.models import Model

from drbcpp.data import DataGenerator, DataMonitor
from drbcpp.evaluation import EvaluateCallback
from drbcpp.loss import pairwise_ranking_crossentropy_loss
from drbcpp.models import drbc_model


class Gym:
    def __init__(self, experiment: str = 'vanilla_drbc'):
        """
        :param experiment: description of the experiment
        """
        self.experiment_path = Path('experiments') / datetime.now().replace(microsecond=0).isoformat()
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        latest = Path('experiments/latest/').absolute()
        latest.unlink(missing_ok=True)
        latest.symlink_to(self.experiment_path.absolute(), target_is_directory=True)

        self.train_generator: DataGenerator = None
        self.valid_generator: DataGenerator = None
        self.model: Model = None

        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')
        self.aim_session = Session(experiment=experiment)

    def construct_datasets(self, min_nodes: int, max_nodes: int, nb_train_graphs: int, nb_valid_graphs: int,
                           graphs_per_batch: int, nb_batches: int,
                           node_neighbors_aggregation: str = 'gcn',
                           graph_type: str = 'powerlaw'):
        """
        :param min_nodes: minimum training scale (node set size)
        :param max_nodes: maximum training scale (node set size)
        :param nb_train_graphs: number of train graphs
        :param nb_valid_graphs: number of validation graphs
        :param graphs_per_batch: number of graphs sampled per batch
        :param nb_batches: number of batches to process per each training epoch
        :param node_neighbors_aggregation: {sum, mean, gcn (weighted sum)}
        :param graph_type: {powerlaw, erdos_renyi, powerlaw, small-world, barabasi_albert}
        """
        self.aim_session.set_params(locals(), name='dataset_params')
        self.train_generator = DataGenerator(tag='Train', graph_type=graph_type, min_nodes=min_nodes, max_nodes=max_nodes, nb_graphs=nb_train_graphs, node_neighbors_aggregation=node_neighbors_aggregation, graphs_per_batch=graphs_per_batch, nb_batches=nb_batches, include_idx_map=False, random_samples=True, log_betweenness=True)
        self.valid_generator = DataGenerator(tag='Valid', graph_type=graph_type, min_nodes=min_nodes, max_nodes=max_nodes, nb_graphs=nb_valid_graphs, node_neighbors_aggregation=node_neighbors_aggregation, graphs_per_batch=1, nb_batches=nb_valid_graphs, include_idx_map=True, random_samples=False, log_betweenness=False)
        return self

    def construct_model(self, optimizer='adam', aggregation: Optional[str] = 'lstm', combine: str = 'gru'):
        """
        :param optimizer: any tf.keras supported optimizer
        :param aggregation: how to aggregate sequences after DrBCRNN {min, max, sum, mean, lstm}
        :param combine: how to combine in each iteration in DrBCRNN {structure2vec, graphsage, gru}
        """
        self.aim_session.set_params(locals(), name='model_params')
        self.model = drbc_model(node_feature_dim=3, aux_feature_dim=4, rnn_repetitions=5,
                                aggregation=aggregation, combine=combine)
        self.model.compile(optimizer=optimizer, loss=pairwise_ranking_crossentropy_loss)
        self.model.summary()
        return self

    def predict(self, gid):
        x, y, idx_map = self.valid_generator[gid]
        result = self.model.predict_on_batch(x=x).flatten()

        # idx_map[i] >= 0:  # corresponds to nodes with 0.0 betw_log value
        result_output = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                         for i, pred_betweenness in enumerate(result)]
        return result_output

    def train(self, epochs: int, stop_patience: int = 5, lr_reduce_patience: int = 2):
        """
        functional API with model.fit doesn't support sparse tensors with the current implementation =>
        we write the training loop ourselves
        """
        self.aim_session.set_params(locals(), name='train_params')
        callbacks = CallbackList([
            EvaluateCallback(self.valid_generator, prepend_str='val_'),
            TensorBoard(self.log_dir, profile_batch=0),
            AimCallback(self.aim_session),
            ModelCheckpoint(self.model_save_path / 'best.h5py', monitor='val_kendal', save_best_only=True, verbose=1, mode='max'),
            EarlyStopping(monitor='val_kendal', patience=stop_patience, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_kendal', patience=lr_reduce_patience, factor=0.7, mode='max'),
            DataMonitor(self.train_generator, self.valid_generator, update_frequency=5, prefetch=1),
        ],  add_history=True, add_progbar=True, verbose=1,
            model=self.model,
            epochs=epochs, steps=len(self.train_generator))

        epoch_logs = {}
        callbacks.on_train_begin()
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            logs = {}
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
        return self.model.history.history


if __name__ == '__main__':
    fire.Fire(Gym)
