from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPool2D, PReLU, Lambda, DepthwiseConv2D, Add, Input, \
    UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import os
import cv2
import numpy as np


def Downsampling1(inp):
    conv = Conv2D(filters=32,
                  kernel_size=(3, 3),
                  strides=(2, 2),
                  padding='same')(inp)

    pool = MaxPool2D(pool_size=(2, 2))(inp)

    out = Concatenate()([conv, pool])

    out = PReLU()(out)

    return out


def DAB_Module(inp):
    inp_split = tf.split(value=inp,
                         num_or_size_splits=2,
                         axis=-1)
    inp1 = inp_split[0]
    inp2 = inp_split[1]

    dw1 = DepthwiseConv2D(kernel_size=(3, 1),
                          padding='same')(inp1)
    dw1 = DepthwiseConv2D(kernel_size=(1, 3),
                          padding='same')(dw1)

    dw2 = DepthwiseConv2D(kernel_size=(3, 1),
                          padding='same',
                          dilation_rate=2)(inp2)
    dw2 = DepthwiseConv2D(kernel_size=(1, 3),
                          padding='same',
                          dilation_rate=2)(dw2)

    dab = Add()([dw1, dw2])

    return dab


def DABNet(shape):
    inp = Input(shape=shape, name="input_tensor")

    out = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding='same')(inp)
    out = PReLU()(out)

    out = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same')(out)
    out = PReLU()(out)

    out = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same')(out)
    out = PReLU()(out)

    down = Lambda(Downsampling1)(out)

    dab = Lambda(DAB_Module, name='DAB_Block_1')(down)

    dab = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 padding='same',
                 data_format='channels_last')(dab)
    dab = PReLU()(dab)

    out = Add()([down, dab])

    out = Conv2D(filters=1,
                 kernel_size=(1, 1),
                 padding='same',
                 activation='sigmoid')(out)

    out = UpSampling2D(size=(4, 4),
                       interpolation='bilinear',
                       name="output_tensor")(out)

    model = Model(inputs=inp,
                  outputs=out)

    return model


def load_model(weight):
    model = DABNet((240, 320, 3))
    # print(model.summary())
    model.load_weights(weight)
    return model
