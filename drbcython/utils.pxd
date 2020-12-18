# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from graph cimport Graph


cdef extern from "../cpp/utils.h":
    cdef cppclass Utils:

        Utils()
        vector[double] Betweenness_Batch(vector[shared_ptr[Graph]] graph_list)
        vector[double] Betweenness(shared_ptr[Graph] graph)
        vector[double] convertToLog(vector[double] CB)
        vector[double] bc_log
        vector[int] bc_bool
