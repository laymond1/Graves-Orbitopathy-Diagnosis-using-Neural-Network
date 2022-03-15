from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import DepthwiseConv2D, Dense, Flatten, Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPool2D, AveragePooling2D, Dropout

class CNN_Model(tf.keras.Model):
    def __init__(self, mode, multiclass=False):
        super(CNN_Model, self).__init__()
        self.mode = mode
        self.multiclass = multiclass

        self._batch_norm_momentum = 0.99
        self._batch_norm_epsilon = 1e-3
        self._channel_axis = -1

        self.filters = 16

        self.max_pool = MaxPool2D(pool_size=[3,3], strides=[2,2], padding='same')
        self.avg_pool = GlobalAveragePooling2D()

        self._build_stem()
        self._build_depthwise()
        self._build_final()

    def _build_stem(self):
        self.ax_stem = Conv2D(
            self.filters, 
            kernel_size=[3,3], 
            strides=[1,1], 
            kernel_initializer='glorot_uniform',
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.ax_stem_bn = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)
        self.co_stem = Conv2D(
            self.filters, 
            kernel_size=[3,3], 
            strides=[1,1], 
            kernel_initializer='glorot_uniform',
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.co_stem_bn = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)
        self.sa_stem = Conv2D(
            self.filters*2, # 32 filters
            kernel_size=[3,3], 
            strides=[1,1], 
            kernel_initializer='glorot_uniform',
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.sa_stem_bn = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)
    
    def _build_depthwise(self):
        self.ax_conv = DepthwiseConv2D(
            kernel_size=[3, 3], 
            strides=[1, 1], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.ax_conv_bn = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)
        self.co_conv = DepthwiseConv2D(
            kernel_size=[3, 3], 
            strides=[1, 1], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.co_conv_bn = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)
        self.sa_conv = DepthwiseConv2D(
            kernel_size=[3, 3], 
            strides=[1, 1], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.sa_conv_bn = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)


        self.ax_l_dw = DepthwiseConv2D(
            kernel_size=[32, 16], 
            strides=[32, 32], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.ax_r_dw = DepthwiseConv2D(
            kernel_size=[32, 16], 
            strides=[32, 32], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.co_l_dw = DepthwiseConv2D(
            kernel_size=[16, 16], 
            strides=[16, 16], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.co_r_dw = DepthwiseConv2D(
            kernel_size=[16, 16], 
            strides=[16, 16], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.sa_l_dw = DepthwiseConv2D(
            kernel_size=[32, 32], 
            strides=[32, 32], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 
        self.sa_r_dw = DepthwiseConv2D(
            kernel_size=[32, 32], 
            strides=[32, 32], 
            kernel_initializer='glorot_uniform',
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.02)) 


    def _build_final(self):
        self.fc_ax = Dense(4, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.fc_co = Dense(4, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.fc_sa = Dense(4, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.fc1 = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.02))
        self.fc2 = Dense(3, activation='softmax', kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.02))

    def call(self, x, training=True):
        if 'ax' in self.mode:
            ax_x = x[:,:,:,:32]
            # Feature map extraction
            ax_x = self.ax_stem_bn(self.ax_stem(ax_x), training=training)
            ax_x = self.max_pool(ax_x)

            ax_x = self.ax_conv_bn(self.ax_conv(ax_x), training=training)
            ax_x = self.max_pool(ax_x)

            # Classification
            ax_l = self.ax_l_dw(ax_x[:,:,:16,:])
            ax_r = self.ax_r_dw(ax_x[:,:,16:,:])

            ax_l = tf.reshape(ax_l, [-1, ax_l.shape[3]])
            ax_r = tf.reshape(ax_r, [-1, ax_r.shape[3]])

            ax_x = tf.concat([ax_l, ax_r], axis=1)
            ax_x = self.fc_ax(ax_x)
            # Only Axial Model
            if self.mode == 'ax':
                x = ax_x
                if self.multiclass :
                    return self.fc2(x)
                else:
                    return self.fc1(x)

        if 'co' in self.mode:
            co_x = x[:,:64,:,32:64]
            # Feature map extraction
            co_x = self.co_stem_bn(self.co_stem(co_x), training=training)
            co_x = self.max_pool(co_x)

            co_x = self.co_conv_bn(self.co_conv(co_x), training=training)
            co_x = self.max_pool(co_x)

            # Classification
            co_l = self.co_l_dw(co_x[:,:,:16,:])
            co_r = self.co_r_dw(co_x[:,:,16:,:])

            co_l = tf.reshape(co_l, [-1, co_l.shape[3]])
            co_r = tf.reshape(co_r, [-1, co_r.shape[3]])

            co_x = tf.concat([co_l, co_r], axis=1)
            co_x = self.fc_co(co_x)
            # Only Coronal Model
            if self.mode == 'co':
                x = co_x
                if self.multiclass:
                    return self.fc2(x)
                else:
                    return self.fc1(x)

        if 'sa' in self.mode:
            sa_x = x[:,:,:,64:]
            # Feature map extraction
            sa_x = self.sa_stem_bn(self.sa_stem(sa_x), training=training)
            sa_x = self.max_pool(sa_x)

            sa_x = self.sa_conv_bn(self.sa_conv(sa_x), training=training)
            sa_x = self.max_pool(sa_x)
            
            # Classification
            sa_l = self.sa_l_dw(sa_x[:,:,:,:16])
            sa_r = self.sa_r_dw(sa_x[:,:,:,16:])

            sa_l = tf.reshape(sa_l, [-1, sa_l.shape[3]])
            sa_r = tf.reshape(sa_r, [-1, sa_r.shape[3]]) 

            sa_x = tf.concat([sa_l, sa_r], axis=1)
            sa_x = self.fc_sa(sa_x)
            # Only Sagittal Model
            if self.mode == 'sa':
                x = sa_x
                if self.multiclass :
                    return self.fc2(x)
                else:
                    return self.fc1(x)
            
        if self.mode == 'axco':
            x = tf.concat([ax_x, co_x], axis=1)
        elif self.mode == 'axsa':
            x = tf.concat([ax_x, sa_x], axis=1)
        elif self.mode == 'cosa':
            x = tf.concat([co_x, sa_x], axis=1)
        else:
            x = tf.concat([ax_x, co_x, sa_x], axis=1)

        if self.multiclass :
            return self.fc2(x)
        else:
            return self.fc1(x) 