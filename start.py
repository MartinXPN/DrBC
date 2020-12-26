from drbcpp.gym import Gym
from drbcpp.util import fix_random_seed


def main():
    # Instructions to load the model later:
    # import tensorflow as tf
    # from tensorflow.keras.models import load_model
    # load_model('path/to/experiments/<DATE>/models/best.h5py', custom_objects={'tf': tf}, compile=False).summary()

    fix_random_seed(42)
    gym = Gym(experiment='vanilla_drbc')
    gym.construct_datasets(min_nodes=400, max_nodes=500,
                           nb_train_graphs=100, nb_valid_graphs=100, graphs_per_batch=4, nb_batches=200,
                           node_neighbors_aggregation='gcn',
                           graph_type='powerlaw')
    gym.construct_model(optimizer='adam', aggregation='max', combine='gru')
    gym.train(epochs=100, stop_patience=5, lr_reduce_patience=2)


if __name__ == "__main__":
    main()
