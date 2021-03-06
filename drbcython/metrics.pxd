# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector


cdef extern from "../cpp/metrics.h":

    cdef cppclass Metrics:
        Metrics() except+
        double MeanSquareError(vector[double] real_data, vector[double] predict_data) except+
        double AvgError(vector[double] real_data, vector[double] predict_data) except+
        double MaxError(vector[double] real_data, vector[double] predict_data) except+
        double RankTopK(vector[double] real_data, vector[double] predict_data,double K) except+
