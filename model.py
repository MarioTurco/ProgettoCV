from turtle import shape
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Reshape, Bidirectional, LSTM, Activation, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

def convolution_block(x,
                      filters,
                      activation,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding="same",
                      pooling_padding="valid",
                      use_bias=True,
                      use_batchnorm=False,
                      use_dropout=False,
                      drop_value=0.2,
                      use_pooling=False,
                      pool_size=(2, 2),
                      pool_stride = None,
                      conv_name = None,
                      batch_name = None
                      ):
    if conv_name != None:
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                        kernel_initializer='he_normal', name=name)(x)
    else:
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                        kernel_initializer='he_normal')(x)
    
    x = activation(x)
    
    if use_batchnorm:
        if batch_name != None:
            x = layers.BatchNormalization(name=batch_name)(x)
        else:
            x = layers.BatchNormalization()(x)

    pool = x
    if use_pooling:
        pool = layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, data_format='channels_last', padding=pooling_padding)(pool)

    if use_dropout:
        pool = layers.Dropout(drop_value)(pool)

    return x, pool

# Build an endpoint layer for implementing CTC loss.
class CTCLayer( layers.Layer ):
    def __init__( self, name=None, **kwargs ):
        super().__init__( name=name )
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it to the layer using `self.add_loss()`.
        batch_len = tf.cast( tf.shape(y_true)[0], dtype='int64' )
        input_length = tf.cast( tf.shape(y_pred)[1], dtype='int64' )
        label_length = y_true.numpy()[:, -1]
        
        input_length = input_length*tf.ones( shape=(batch_len,1), dtype='int64' )
        #label_length = label_length*tf.ones( shape=(batch_len,1), dtype='int64' )

        loss = self.loss_fn( y_true[:, :-1], y_pred, input_length, label_length )
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = y_true.numpy()[:, -1]

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    loss = K.ctc_batch_cost(y_true[:, :-1], y_pred, input_length, label_length)
    return loss


def custom_ctc():
    """Custom CTC loss implementation"""

    def loss(y_true, y_pred):
        """Why you make it so complicated?
        
        Since the prediction from models is (batch, timedistdim, tot_num_uniq_chars)
        and the true target is labels (batch_size,1) but the ctc loss need some
        additional information of different sizes. And the inputs to loss y_true,
        y_pred must be both same dimensions because of keras.
        
        So I have packed the needed information inside the y_true and just made it
        to a matching dimension with y_true"""

        batch_labels = y_true[:, :-2]
        label_length = y_true[:, -2]
        input_length = y_true[:, -1]
        
        
        #reshape for the loss, add that extra meaningless dimension
        label_length = tf.expand_dims(label_length, -1)
        input_length = tf.expand_dims(input_length, -1)


        return K.ctc_batch_cost(batch_labels, y_pred, input_length, label_length)
    return loss


def build_and_compile_model_original(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, _ = convolution_block(imgs, 64, Activation('relu'), use_pooling=True)
    x, _ = convolution_block(x, 128, Activation('relu'), use_pooling=True)
    x, _ = convolution_block(x, 256, Activation('relu'))
    x, _ = convolution_block(x, 256, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 512, Activation('relu'), use_batchnorm=True)
    x, _ = convolution_block(x, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 512, Activation('relu'), kernel_size=(2, 2), padding="valid")
    
    conv_shape = x.get_shape()

    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = Dense(64, activation="relu", use_bias=True)(x)

    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, p1 = convolution_block(imgs, 32, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p2 = convolution_block(p1, 64, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p2, 128, Activation('relu'))
    x, _ = convolution_block(x, 128, Activation('relu'))
    x, p3 = convolution_block(x, 128, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p4 = convolution_block(p3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p4, 256, Activation('relu'), kernel_size=(2, 2))
    
    tdist = TimeDistributed(Flatten())(x)
    
    x = Dense(64, activation="relu", use_bias=True)(tdist)

    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model


def build_and_compile_model_v3(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, p1 = convolution_block(imgs, 32, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p2 = convolution_block(p1, 64, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 128, Activation('relu'))
    x, p3 = convolution_block(x, 128, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p3, 128, Activation('relu'))
    x, p4 = convolution_block(p3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p4, 256, Activation('relu'), kernel_size=(2, 2))
    
    tdist = TimeDistributed(Flatten())(x)
    
    x = Dense(64, activation="relu", use_bias=True)(tdist)
    x = Dropout( 0.25)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v4(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, p1 = convolution_block(imgs, 64, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p2 = convolution_block(p1, 128, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 256, Activation('relu'))
    x, p3 = convolution_block(x, 256, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p3, 256, Activation('relu'))
    x, p4 = convolution_block(p3, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p4, 512, Activation('relu'), kernel_size=(2, 2))
    
    tdist = TimeDistributed(Flatten())(x)
    
    x = Dense(64, activation="relu", use_bias=True)(tdist)
    x = Dropout( 0.25)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v5(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, p1 = convolution_block(imgs, 64, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p2 = convolution_block(p1, 128, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 256, Activation('relu'))
    x, p3 = convolution_block(x, 256, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p3, 256, Activation('relu'), use_batchnorm=True)
    x, p4 = convolution_block(p3, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p4, 512, Activation('relu'), kernel_size=(2, 2))
    
    tdist = TimeDistributed(Flatten())(x)
    
    x = Dense(64, activation="relu", use_bias=True)(tdist)
    x = Dropout( 0.25)(x)
    
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0, dropout=0.25))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0, dropout=0.25))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v6(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, p1 = convolution_block(imgs, 64, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p2 = convolution_block(p1, 128, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 256, Activation('relu'))
    x, p3 = convolution_block(x, 256, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p3, 256, Activation('relu'), use_batchnorm=True)
    x, p4 = convolution_block(p3, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p4, 512, Activation('relu'), kernel_size=(2, 2))
    
    tdist = TimeDistributed(Flatten())(x)
    
    x = Dense(64, activation="relu", use_bias=True)(tdist)
    x = Dropout( 0.25)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model


def build_and_compile_model_v7(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 64, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 128, Activation('relu'), use_batchnorm=True)
    x2 = Concatenate()([x2, p1])
    _, p2 = convolution_block(x2, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 256, Activation('relu'), use_batchnorm=True)
    x3 = Concatenate()([x3, p2])
    _, p3 = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x4, _ = convolution_block(p3, 512, Activation('relu'), use_batchnorm=True)
    x4 = Concatenate()([x4, p3])
    _, p4 = convolution_block(x4, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p4)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model
    
def build_and_compile_model_v8(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    #labels = Input(shape=(None,))

    x, p1 = convolution_block(imgs, 64, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, p2 = convolution_block(p1, 128, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(x, 256, Activation('relu'))
    x, p3 = convolution_block(p2, 256, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p3, 256, Activation('relu'), use_batchnorm=True)
    x, p4 = convolution_block(x, 512, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x, _ = convolution_block(p4, 512, Activation('relu'), kernel_size=(2, 2))
    
    tdist = TimeDistributed(Flatten())(x)
    
    x = Dense(64, activation="relu", use_bias=True)(tdist)
    x = Dropout(0.25)(x)
    
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0, dropout=0.25))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0, dropout=0.25))(x)

    output = Dense(len_characters+1, activation='softmax')(x)

    #output = CTCLayer()(labels, x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model


def build_and_compile_model_v9(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 16, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 16, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 32, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 32, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 64, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_2(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 16, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 16, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 32, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 32, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 64, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_3(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 32, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 64, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 128, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_4(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 32, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 64, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 128, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_5(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 32, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 64, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 128, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 256, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_6(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 64, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 128, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 256, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 512, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_7(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 64, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 128, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 256, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 512, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v9_8(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 64, Activation('relu'), use_batchnorm=True)
    _, p1 = convolution_block(x1, 64, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x2, _ = convolution_block(p1, 128, Activation('relu'), use_batchnorm=True)
    _, p2 = convolution_block(x2, 128, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 256, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x3, _ = convolution_block(p2, 512, Activation('relu'), use_batchnorm=True)
    _, p3 = convolution_block(x3, 1024, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    
    tdist = TimeDistributed(Flatten())(p3)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v10(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 8, Activation('relu'), use_batchnorm=True)
    x2, _ = convolution_block(x1, 16, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x2, 32, Activation('relu'), use_batchnorm=True)
    x4, _ = convolution_block(x3, 64, Activation('relu'), use_batchnorm=True)
    
    combined = Concatenate()([x1, x2, x3, x4])
    
    tdist = TimeDistributed(Flatten())(combined)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model


def build_and_compile_model_v10_2(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 16, Activation('relu'), use_batchnorm=True)
    x2, _ = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True)
    x4, _ = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True)
    
    combined = Concatenate()([x1, x2, x3, x4])
    
    tdist = TimeDistributed(Flatten())(combined)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v10_3(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 32, Activation('relu'), use_batchnorm=True)
    x2, _ = convolution_block(x1, 64, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x2, 128, Activation('relu'), use_batchnorm=True)
    x4, _ = convolution_block(x3, 256, Activation('relu'), use_batchnorm=True)
    
    combined = Concatenate()([x1, x2, x3, x4])
    
    tdist = TimeDistributed(Flatten())(combined)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v10_4(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 16, Activation('relu'), use_batchnorm=True)
    x2, _ = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True)
    x4, _ = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True)
    x5, _ = convolution_block(x4, 256, Activation('relu'), use_batchnorm=True)
    
    combined = Concatenate()([x1, x2, x3, x4, x5])
    
    tdist = TimeDistributed(Flatten())(combined)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v10_5(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 16, Activation('relu'), use_batchnorm=True)
    x2, _ = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True)
    x4, _ = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True)
    x5, _ = convolution_block(x4, 256, Activation('relu'), use_batchnorm=True)
    
    combined = Concatenate()([x1, x2, x3, x4, x5])
    combined, _ = convolution_block(combined, 8, Activation('relu'), use_batchnorm=True)
    
    tdist = TimeDistributed(Flatten())(combined)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model

def build_and_compile_model_v10_6(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    
    x1, _ = convolution_block(imgs, 16, Activation('relu'), use_batchnorm=True)
    x2, _ = convolution_block(x1, 32, Activation('relu'), use_batchnorm=True)
    x3, _ = convolution_block(x2, 64, Activation('relu'), use_batchnorm=True)
    x4, _ = convolution_block(x3, 128, Activation('relu'), use_batchnorm=True)
    x5, _ = convolution_block(x4, 256, Activation('relu'), use_batchnorm=True)
    
    combined = Concatenate()([x1, x2, x3, x4, x5])
    combined, _ = convolution_block(combined, 256, Activation('relu'), use_batchnorm=True)
    
    tdist = TimeDistributed(Flatten())(combined)
    
    x = Dense(128, activation="relu", use_bias=True)(tdist)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0))(x)
    
    output = Dense(len_characters+1, activation='softmax')(x)

    model = Model(inputs=imgs, outputs=[output])

    model.compile(optimizer=opt, loss=custom_ctc())

    return model