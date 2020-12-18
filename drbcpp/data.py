from dataclasses import dataclass, field
from typing import List, Dict

import networkx as nx
import numpy as np
from drbcython.graph import py_GSet
from drbcython.utils import py_Utils
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from drbcython import utils, graph, PrepareBatchGraph


@dataclass
class DataGenerator(Sequence):
    utils: py_Utils = field(default_factory=lambda: utils.py_Utils())
    graphs: py_GSet = field(default_factory=lambda: graph.py_GSet())
    count: int = 0
    betweenness: List[float] = field(default_factory=list)

    tag: str = 'generator'
    graph_type: str = ''
    min_nodes: int = 0
    max_nodes: int = 0
    nb_graphs: int = 1
    graphs_per_batch: int = 1
    nb_batches: int = 1
    node_neighbors_aggregation: str = 'gcn'
    include_idx_map: bool = False
    random_samples: bool = True
    log_betweenness: bool = True
    compute_betweenness: bool = True
    neighbor_aggregation_ids: Dict[str, int] = field(default_factory=lambda: {'sum': 0, 'mean': 1, 'gcn': 2})

    def __len__(self) -> int:
        return self.nb_batches

    def get_batch(self, graphs, ids: List[int]):
        label = []
        for i in ids:
            label += self.betweenness[i]
        label = np.array(label)

        aggregator_id = self.neighbor_aggregation_ids[self.node_neighbors_aggregation]
        batch_graph = PrepareBatchGraph.py_PrepareBatchGraph(aggregator_id)
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
