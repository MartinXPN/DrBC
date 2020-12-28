import pickle as cp
import time

import fire
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tqdm import trange, tqdm

from drbc.gym import DataGenerator, EvaluateCallback


def evaluate_synthetic_data(data_test, model_path):
    """ This function is most probably wrong because the synthetic data is one file per graph and score """
    model = load_model(model_path, custom_objects={'tf': tf}, compile=False)
    data = DataGenerator(tag='Synthetic', include_idx_map=True, random_samples=False, compute_betweenness=False)
    evaluate = EvaluateCallback(data, prepend_str='')
    evaluate.set_model(model)

    with open(data_test, 'rb') as f:
        valid_data = cp.load(f)
    graph_list = valid_data[0]
    betweenness_list = valid_data[1]

    for i in trange(100):
        g = graph_list[i]
        data.add_graph(g)
    data.betweenness = betweenness_list
    data.clear()

    res = {}
    evaluate.on_epoch_end(0, res)
    return res


def evaluate_real_data(model_path, data_test, label_file):
    model: Model = load_model(model_path, custom_objects={'tf': tf}, compile=False)
    data = DataGenerator(tag='RealData', include_idx_map=True, random_samples=False, compute_betweenness=False)
    evaluator = EvaluateCallback(data, prepend_str='')
    evaluator.set_model(model)

    exact_betweenness = []
    with open(label_file) as f:
        for line in tqdm(f):
            exact_betweenness.append(float(line.strip().split()[1]))

    g = nx.read_weighted_edgelist(data_test)

    start = time.time()
    data.add_graph(g)
    data.betweenness = [exact_betweenness]
    res = evaluator.evaluate()
    end = time.time()

    res['run_time'] = end - start
    data.clear()
    return res


if __name__ == '__main__':
    fire.Fire({
        'synthetic': evaluate_synthetic_data,
        'real': evaluate_real_data,
    })
