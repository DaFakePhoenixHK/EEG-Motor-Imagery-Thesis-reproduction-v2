"""
Copyright (C) 2022 King Saud University, Saudi Arabia
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Author: Hamdi Altaheri
"""

import math
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Conv1D
from tensorflow.keras.layers import multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import backend as K


def attention_block(in_layer, attention_model, ratio=8, residual=False, apply_to_input=True):
    in_sh = in_layer.shape
    in_len = len(in_sh)
    expanded_axis = 2

    if attention_model == 'mha':
        if in_len > 3:
            in_layer = Reshape((in_sh[1], -1))(in_layer)
        out_layer = mha_block(in_layer)
    elif attention_model == 'mhla':
        if in_len > 3:
            in_layer = Reshape((in_sh[1], -1))(in_layer)
        out_layer = mha_block(in_layer, vanilla=False)
    elif attention_model == 'se':
        if in_len < 4:
            in_layer = tf.expand_dims(in_layer, axis=expanded_axis)
        out_layer = se_block(in_layer, ratio, residual, apply_to_input)
    elif attention_model == 'cbam':
        if in_len < 4:
            in_layer = tf.expand_dims(in_layer, axis=expanded_axis)
        out_layer = cbam_block(in_layer, ratio=ratio, residual=residual)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_model))

    if in_len == 3 and len(out_layer.shape) == 4:
        out_layer = tf.squeeze(out_layer, expanded_axis)
    elif in_len == 4 and len(out_layer.shape) == 3:
        out_layer = Reshape((in_sh[1], in_sh[2], in_sh[3]))(out_layer)
    return out_layer


def mha_block(input_feature, key_dim=8, num_heads=2, dropout=0.5, vanilla=True):
    """Multi Head self Attention (MHA) block."""
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    if vanilla:
        x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)
    else:
        NUM_PATCHES = input_feature.shape[1]
        diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
        x = MultiHeadAttention_LSA(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(
            x, x, attention_mask=diag_attn_mask)
    x = Dropout(0.3)(x)
    mha_feature = Add()([input_feature, x])
    return mha_feature


class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    """Locality Self Attention (LSA)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores


def se_block(input_feature, ratio=8, residual=False, apply_to_input=True):
    """Squeeze-and-Excitation (SE) block."""
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    if ratio != 0:
        se_feature = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal',
                          use_bias=True, bias_initializer='zeros')(se_feature)
        se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal',
                          use_bias=True, bias_initializer='zeros')(se_feature)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)
    if apply_to_input:
        se_feature = multiply([input_feature, se_feature])
    if residual:
        se_feature = Add()([se_feature, input_feature])
    return se_feature


def cbam_block(input_feature, ratio=8, residual=False):
    """Convolutional Block Attention Module (CBAM)."""
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    if residual:
        cbam_feature = Add()([input_feature, cbam_feature])
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])


def eca_attention(input_feature, gama=2, b=1):
    """Efficient Channel Attention (ECA). From DB-ATCNet: https://github.com/zk-xju/DB-ATCNet"""
    in_channel = input_feature.shape[-1]
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    x = GlobalAveragePooling2D()(input_feature)
    x = Reshape(target_shape=(in_channel, 1))(x)
    x = Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, in_channel))(x)
    return multiply([input_feature, x])


def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        cbam_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                          activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])
