from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense, LeakyReLU, LSTM, Attention, MultiHeadAttention
from tensorflow.keras.models import Model

from drbc.layers import DrBCRNN


def drbc_model(node_feature_dim: int = 3, aux_feature_dim: int = 4, rnn_repetitions: int = 5,
               aggregation: Optional[str] = 'max', combine: str = 'gru'):
    """
    @param node_feature_dim: initial node features, [Dc,1,1]
    @param aux_feature_dim: extra node features in the hidden layer in the decoder, [Dc,CI1,CI2,1]
    @param rnn_repetitions: how many loops are there in DrBCRNN
    @param aggregation: how to aggregate sequences after DrBCRNN {min, max, sum, mean, multi_attention, lstm}
    @param combine: how to combine in each iteration in DrBCRNN {structure2vec, graphsage, gru}
    @return: DrBC tf.keras model
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
    elif aggregation == 'multi_attention':
        n2n_features = MultiHeadAttention(num_heads=4, key_dim=128, dropout=0.2)(n2n_features, n2n_features)
        n2n_features = Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='aggregate')(n2n_features)
    elif aggregation == 'lstm':
        n2n_features = LSTM(units=128, return_sequences=True)(n2n_features)
        n2n_features = Attention(use_scale=True)([n2n_features, n2n_features, n2n_features])
        n2n_features = Lambda(lambda x: tf.reduce_sum(x, axis=-1), name='aggregate')(n2n_features)
    n2n_features = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalize_n2n')(n2n_features)

    all_features = Concatenate(axis=-1)([n2n_features, input_aux_features])
    top = Dense(64)(all_features)
    top = LeakyReLU()(top)
    out = Dense(1)(top)

    return Model(inputs=[input_node_features, input_aux_features, input_n2n], outputs=out, name='DrBC')
