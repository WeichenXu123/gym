from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from keras import backend as K
from keras.engine.base_layer import Layer

from keras.layers import Lambda

import backend2 as K2

_NEG_INF = -1e9


def get_padding(x):
    return K.cast(K.equal(x, 0), 'float32')


def padding_layer():
    return Lambda(lambda x: get_padding(x))


def get_padding_bias(x):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF
    return K.expand_dims(K.expand_dims(attention_bias, axis=1), axis=1)


def padding_bias(x):
    return Lambda(lambda x: get_padding_bias(x))(x)


def get_position_encoding(
        length, # this is tensor
        hidden_size,
        min_timescale=1.0,
        max_timescale=1.0e4):
    position = K.cast(K2.range(0, length), 'float32')
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1.0))
    inv_timescales = min_timescale * K.exp(
        K.cast(K2.range(0, num_timescales), 'float32') * (-log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return signal


def with_position_encoding(
        embedding_input,
        hidden_size,
        min_timescale=1.0,
        max_timescale=1.0e4):
    def with_position_encoding_func(embedded_inputs):
        length = K.shape(embedded_inputs)[1]
        return embedded_inputs + get_position_encoding(
            length, hidden_size, min_timescale, max_timescale)

    return Lambda(with_position_encoding_func)(embedding_input)


def get_decoder_self_attention_bias(length):
    valid_locs = K2.matrix_band_part(K.ones([length, length]), -1, 0)
    valid_locs = K.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def decoder_self_attention_bias_layer():
    return Lambda(lambda length: get_decoder_self_attention_bias(length))


class LayerNormalization(Layer):

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        shape = [hidden_size]
        self.scale = K.variable(K.ones(shape=shape, dtype='float32'), dtype='float32')
        self.bias = K.variable(K.zeros(shape=shape, dtype='float32'), dtype='float32')

    def call(self, x, epsilon=1e-6):
        mean = K.mean(x, axis=[-1], keepdims=True)
        variance = K.mean(K.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * K.pow(variance + epsilon, -0.5)
        return norm_x * self.scale + self.bias
