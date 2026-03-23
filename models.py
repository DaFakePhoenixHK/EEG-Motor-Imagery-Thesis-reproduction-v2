"""
Copyright (C) 2022 King Saud University, Saudi Arabia
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Author: Hamdi Altaheri
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from attention_models import attention_block, eca_attention


def ATCNet_(n_classes, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', fuse='average'):
    """ATCNet model from Altaheri et al 2023. https://ieeexplore.ieee.org/document/9852687"""
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)
    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    from_logits = False
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    sw_concat = []
    # For static shape: block1 (batch, time, F2). Default in_samples=1125 -> time=20 after conv+pool.
    seq_len = block1.shape[1]
    if seq_len is None:
        seq_len = 20  # fallback for BCI2a default architecture
    for i in range(n_windows):
        st = i
        end = seq_len - n_windows + i + 1
        block2 = Lambda(lambda x, a=st, b=end: x[:, a:b, :])(block1)

        if attention is not None:
            if attention == 'se' or attention == 'cbam':
                block2 = Permute((2, 1))(block2)
                block2 = attention_block(block2, attention)
                block2 = Permute((2, 1))(block2)
            else:
                block2 = attention_block(block2, attention)

        block3 = TCN_block_(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))
        elif fuse == 'concat':
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if fuse == 'average':
        if len(sw_concat) > 1:
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:
            sw_concat = sw_concat[0]
    elif fuse == 'concat':
        sw_concat = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(sw_concat)

    if from_logits:
        out = Activation('linear', name='linear')(sw_concat)
    else:
        out = Activation('softmax', name='softmax')(sw_concat)
    return Model(inputs=input_1, outputs=out)


def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False, depth_multiplier=D,
                             data_format='channels_last', depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1), data_format='channels_last', use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay=0.009, maxNorm=0.6, dropout=0.25):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), depth_multiplier=D, data_format='channels_last',
                             depthwise_regularizer=L2(weightDecay),
                             depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1), data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]), use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)
    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out


def TCN_block_(input_layer, input_dimension, depth, kernel_size, filters, dropout,
               weightDecay=0.009, maxNorm=0.6, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay), kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(weightDecay), kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1, kernel_regularizer=L2(weightDecay),
                      kernel_constraint=max_norm(maxNorm, axis=[0, 1]), padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)
    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                       kernel_regularizer=L2(weightDecay), kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                       kernel_regularizer=L2(weightDecay), kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out


def TCNet_Fusion(n_classes, Chans=22, Samples=1125, layers=2, kernel_s=4, filt=12,
                 dropout=0.3, activation='elu', F1=24, D=2, kernLength=32, dropout_eeg=0.3):
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = 0.25
    F2 = F1 * D
    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    FC = Flatten()(block2)
    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers, kernel_size=kernel_s,
                     filters=filt, dropout=dropout, activation=activation)
    Con1 = Concatenate()([block2, outs])
    out = Flatten()(Con1)
    Con2 = Concatenate()([out, FC])
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(Con2)
    softmax = Activation('softmax', name='softmax')(dense)
    return Model(inputs=input1, outputs=softmax)


def EEGTCNet(n_classes, Chans=22, Samples=1125, layers=2, kernel_s=4, filt=12, dropout=0.3,
             activation='elu', F1=8, D=2, kernLength=32, dropout_eeg=0.2):
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = 0.25
    F2 = F1 * D
    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers, kernel_size=kernel_s,
                     filters=filt, dropout=dropout, activation=activation)
    out = Lambda(lambda x: x[:, -1, :])(outs)
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(out)
    softmax = Activation('softmax', name='softmax')(dense)
    return Model(inputs=input1, outputs=softmax)


def MBEEG_SENet(nb_classes, Chans, Samples, D=2):
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = 0.25
    EEGNet_sep1 = EEGNet(input_layer=input2, F1=4, kernLength=16, D=D, Chans=Chans, dropout=0)
    EEGNet_sep2 = EEGNet(input_layer=input2, F1=8, kernLength=32, D=D, Chans=Chans, dropout=0.1)
    EEGNet_sep3 = EEGNet(input_layer=input2, F1=16, kernLength=64, D=D, Chans=Chans, dropout=0.2)
    SE1 = attention_block(EEGNet_sep1, 'se', ratio=4)
    SE2 = attention_block(EEGNet_sep2, 'se', ratio=4)
    SE3 = attention_block(EEGNet_sep3, 'se', ratio=2)
    FC1 = Flatten()(SE1)
    FC2 = Flatten()(SE2)
    FC3 = Flatten()(SE3)
    CON = Concatenate()([FC1, FC2, FC3])
    dense1 = Dense(nb_classes, name='dense1', kernel_constraint=max_norm(regRate))(CON)
    softmax = Activation('softmax', name='softmax')(dense1)
    return Model(inputs=input1, outputs=softmax)


def EEGNeX_8_32(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))
    model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias=False,
                             depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(AveragePooling2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 2),
                    data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 4),
                    data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def EEGNet_classifier(n_classes, Chans=22, Samples=1125, F1=8, D=2, kernLength=64, dropout_eeg=0.25):
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = 0.25
    eegnet = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    eegnet = Flatten()(eegnet)
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(eegnet)
    softmax = Activation('softmax', name='softmax')(dense)
    return Model(inputs=input1, outputs=softmax)


def EEGNet(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D,
                             data_format='channels_last', depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1), data_format='channels_last', use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def DeepConvNet(nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
    input_main = Input((1, Chans, Samples))
    input_2 = Permute((2, 3, 1))(input_main)
    block1 = Conv2D(25, (1, 10), input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_2)
    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(50, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropoutRate)(block2)
    block3 = Conv2D(100, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropoutRate)(block3)
    block4 = Conv2D(200, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropoutRate)(block4)
    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    return Model(inputs=input_main, outputs=softmax)


def DB_ATCNet(n_classes, in_chans=22, in_samples=1125, n_windows=3, attention='mha',
              eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=8, eegn_dropout=0.3,
              tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
              tcn_activation='elu', fuse='average', drop1=0.35, drop2=0.1, drop3=0.15, drop4=0.15,
              depth1=2, depth2=4):
    """DB-ATCNet from https://github.com/zk-xju/DB-ATCNet (Dual-Branch + ECA attention)."""
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)
    regRate = 0.25
    F2 = eegn_F1 * eegn_D

    block1 = _ADBC(input_layer=input_2, F1=eegn_F1, D=eegn_D, kernLength=eegn_kernelSize,
                   poolSize=eegn_poolSize, in_chans=in_chans, dropout=eegn_dropout,
                   drop1=drop1, depth1=depth1, depth2=depth2)
    block1 = eca_attention(block1)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    sw_concat = []
    for i in range(n_windows):
        st, end = i, block1.shape[1] - n_windows + i + 1
        block2 = Lambda(lambda x, a=st, b=end: x[:, a:b, :])(block1)
        block2 = attention_block(block2, 'mha')
        block3 = _TCFN_DB(block2, input_dimension=F2, depth=tcn_depth, kernel_size=tcn_kernelSize,
                         filters=tcn_filters, dropout=tcn_dropout, activation=tcn_activation,
                         drop2=drop2, drop3=drop3, drop4=drop4)
        block3 = Lambda(lambda x: x[:, -1, :])(block3)
        sw_concat.append(Dense(n_classes, kernel_constraint=max_norm(regRate))(block3))

    sw_concat = tf.keras.layers.Average()(sw_concat)
    softmax = Activation('softmax', name='softmax')(sw_concat)
    return Model(inputs=input_1, outputs=softmax)


def _ADBC(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1,
          drop1=0.3, depth1=2, depth2=4):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = eca_attention(block1)
    block2 = DepthwiseConv2D((1, in_chans), depth_multiplier=depth1, use_bias=False,
                             data_format='channels_last', depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1), data_format='channels_last', use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    block4 = DepthwiseConv2D((1, in_chans), depth_multiplier=depth2, use_bias=False,
                             data_format='channels_last', depthwise_constraint=max_norm(1.))(block1)
    block4 = BatchNormalization(axis=-1)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D((8, 1), data_format='channels_last')(block4)
    block4 = Dropout(dropout)(block4)
    block5 = Conv2D(F2, (16, 1), data_format='channels_last', use_bias=False, padding='same')(block4)
    block5 = BatchNormalization(axis=-1)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block5)
    block5 = Dropout(dropout)(block5)
    out = Add()([block3, block5])
    out = Dropout(drop1)(out)
    return out


def _TCFN_DB(input_layer, input_dimension, depth, kernel_size, filters, dropout,
             drop2=0.1, drop3=0.15, drop4=0.15, activation='elu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        inp_d = Dropout(drop2)(input_layer)
        added = Add()([block, inp_d])
    out = Activation(activation)(added)
    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                      padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        inp_d = Dropout(drop3)(input_layer)
        block = Add()([block, inp_d])
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                      padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        inp_d = Dropout(drop4)(input_layer)
        added = Add()([block, inp_d])
        out = Activation(activation)(added)
    return out


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    input_main = Input((1, Chans, Samples))
    input_2 = Permute((2, 3, 1))(input_main)
    block1 = Conv2D(40, (1, 25), input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_2)
    block1 = Conv2D(40, (Chans, 1), use_bias=False, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    return Model(inputs=input_main, outputs=softmax)
