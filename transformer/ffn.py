from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine.base_layer import Layer
from keras.layers import Dense, Dropout
from model_utils import WeightsRef

class FeedFowardNetwork(Layer):

    def __init__(self,
                 hidden_size,
                 filter_size,
                 relu_dropout,
                 **kwargs):

        super(FeedFowardNetwork, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.filter_dense_layer = Dense(filter_size, use_bias=True, activation='relu')
        self.output_dense_layer = Dense(hidden_size, use_bias=True)

    # def compute_output_shape(self, input_shape):
    #    return input_shape[0]

    # TODO: add padding support
    def call(self, inputs, train=True, **kwargs):
        # assert isinstance(inputs, (list, tuple))
        # inputs = inputs[0]

        # batch_size = K.shape(inputs)[0]
        # length = K.shape(inputs)[1]

        output = self.filter_dense_layer(inputs)
        if train:
            output = K.dropout(output, self.relu_dropout)
        output = self.output_dense_layer(output)
        return output

    # TODO: add get_config/from_config


def feed_forward_network(x, params):
    filter_dense_layer = Dense(params.filter_size, use_bias=True, activation='relu')
    dropout_layer = Dropout(params.relu_dropout)
    output_dense_layer = Dense(params.hidden_size, use_bias=True)
    output = filter_dense_layer(x)
    output = dropout_layer(output)
    output = output_dense_layer(output)

    wr = WeightsRef()
    wr.filter_dense_layer = filter_dense_layer
    wr.output_dense_layer = output_dense_layer
    return output, wr
