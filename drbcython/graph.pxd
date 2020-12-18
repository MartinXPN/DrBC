# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.map cimport map
from libcpp.pair cimport pair


cdef extern from "../src/lib/graph.h":
    cdef cppclass Graph:
        Graph()except+
        Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to) except+
        int num_nodes
        int num_edges
        vector[vector[int]] adj_list
        vector[pair[int,int]] edge_list

cdef extern from "../src/lib/graph.h":
    cdef cppclass GSet:
        GSet()except+
        void InsertGraph(int gid, shared_ptr[Graph] graph)except+
        vector[shared_ptr[Graph]] Sample_Batch(int batch_size)
        int GetSampleID()except+
        shared_ptr[Graph] Get(int gid)except+
        void Clear()except+
        map[int, shared_ptr[Graph]] graph_pool

