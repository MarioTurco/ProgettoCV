from turtle import shape
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Reshape, Bidirectional, LSTM, Activation
from keras.optimizers import Adam


def convolution_block(x,
                      filters,
                      activation,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding="same",
                      use_bias=True,
                      use_batchnorm=False,
                      use_dropout=False,
                      drop_value=0.2,
                      use_pooling=False,
                      pool_size=(2, 2),
                      pool_stride = (2, 2),
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
        pool = layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, data_format='channels_last')(pool)

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
        label_length = tf.cast( tf.shape(y_true)[1], dtype='int64' )
        
        input_length = input_length*tf.ones( shape=(batch_len,1), dtype='int64' )
        label_length = label_length*tf.ones( shape=(batch_len,1), dtype='int64' )

        loss = self.loss_fn( y_true, y_pred, input_length, label_length )
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_and_compile_model(input_shape, len_characters, opt=Adam()):
    imgs = Input(shape=input_shape)
    labels = Input(shape=(None,), dtype='float32', name="Label" )

    x = convolution_block(imgs, 64, Activation('relu'), use_pooling=True)
    x = convolution_block(x, 128, Activation('relu'), use_pooling=True)
    x = convolution_block(x, 256, Activation('relu'))
    x = convolution_block(x, 256, Activation('relu'), use_pooling=True, pool_size=(1, 2))
    x = convolution_block(x, 512, Activation('relu'), use_batchnorm=True)
    x = convolution_block(x, 512, Activation('relu'), use_batchnorm=True, use_pooling=True, pool_size=(1, 2))
    x = convolution_block(x, 512, Activation('relu'), kernel_size=(2, 2), padding="valid")
    
    conv_shape = x.get_shape()

    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = Dense(64, activation="relu", use_bias=True)(x)

    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0))(x)

    x = Dense(len_characters+1, activation='softmax')(x)

    output = CTCLayer()(labels, x)

    model = Model(inputs=[imgs, labels], outputs=[output])

    model.compile(optimizer=opt)

    return model
    
