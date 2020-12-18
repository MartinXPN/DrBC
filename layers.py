import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, GRU, Dense, Add, Concatenate


@tf.keras.utils.register_keras_serializable(package='drbc', name='GraphSage')
class GraphSage(Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.fc1 = Dense(units=units, activation='relu')
        self.concat = Concatenate()
        self.fc2 = Dense(units=units, activation='relu')

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.concat([x, inputs])
        x = self.fc2(x)
        return x

    def get_config(self):
        return {'units': self.units}


@tf.keras.utils.register_keras_serializable(package='drbc', name='DrBCRNN')
class DrBCRNN(Layer):
    def __init__(self, units=128, repetitions=5, combine='gru', return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.repetitions = repetitions
        self.combine_method = combine
        self.return_sequences = return_sequences

        combine = combine.strip().lower()
        if combine == 'graphsage':          self.combine = GraphSage(units=units, name='graphsage')
        elif combine == 'structure2vec':    self.combine = Add(name='structure2vec')
        elif combine == 'gru':              self.combine = GRU(units=units, return_sequences=False, name='gru')
        else:                               raise ValueError(f'Combine method `{combine}` is not implemented yet!')

        self.node_linear = Dense(self.units)

    def call(self, inputs, **kwargs):
        n2n, message = inputs
        states = [message]
        for rep in range(self.repetitions):
            n2n_pool = tf.sparse.sparse_dense_matmul(n2n, states[rep])
            # print(n2n_pool)
            node_representations = self.node_linear(n2n_pool)
            if self.combine_method == 'graphsage':          combined = self.combine(node_representations)
            elif self.combine_method == 'structure2vec':    combined = self.combine([node_representations, message])
            elif self.combine_method == 'gru':              combined = self.combine(tf.expand_dims(node_representations, 1))
            else:                                           raise ValueError(f'Combine method `{self.combine_method}` is not implemented yet!')
            res = K.l2_normalize(combined, axis=1)
            states.append(res)

        if not self.return_sequences:
            return states[-1]

        # B x embeding_dim x repetitions
        target_shape = [dim if dim is not None else -1 for dim in K.int_shape(message)]
        target_shape.append(self.repetitions)
        out = tf.concat(states[1:], axis=-1)
        out = tf.reshape(out, shape=target_shape)
        return out

    def get_config(self):
        return {
            'units': self.units,
            'repetitions': self.repetitions,
            'combine': self.combine_method,
            'return_sequences': self.return_sequences,
        }
