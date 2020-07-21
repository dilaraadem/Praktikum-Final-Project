import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
import numpy as np
tfd = tf.contrib.distributions
tf.enable_eager_execution()


# ENCODER
# integer num_layers, at least 3, at most 5
# input_dimension: size of original initial data
# latent_dimension: integer, dimension of latent space
# activation: activation function.
# h1, h2, h3, h4, h5: integer, hidden layer dimensions. h3 and h4 are optional.
# num_layers: integer, number of layers. at least 3, at most 5.
# dropout: optional, boolean, if dropout to be used or not
def encoder(input_dimension, latent_dimension, activation, h1, h2, h5, h3=80, h4=50, num_layers=4, dropout=False):
    input_data = tf.keras.Input(shape=(input_dimension,)) # define input
    # dense layer 1
    encoded = tf.keras.layers.Dense(units=h1, activation=activation)(input_data)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    if dropout:
        encoded = tf.keras.layers.Dropout(0.3)(encoded)
    # dense layer 2
    encoded = tf.keras.layers.Dense(units=h2, activation=activation)(encoded)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    if dropout:
        encoded = tf.keras.layers.Dropout(0.3)(encoded)
    if num_layers>3:
        # dense layer 3
        encoded = tf.keras.layers.Dense(units=h3, activation=activation)(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        if dropout:
            encoded = tf.keras.layers.Dropout(0.3)(encoded)
    if num_layers>4:
        # dense layer 4
        encoded = tf.keras.layers.Dense(units=h4, activation=activation)(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        if dropout:
            encoded = tf.keras.layers.Dropout(0.3)(encoded)
    # dense layer 5
    encoded = tf.keras.layers.Dense(units=h5, activation=activation)(encoded)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    if dropout:
        encoded = tf.keras.layers.Dropout(0.3)(encoded)
    # latent space
    encoded = tf.keras.layers.Dense(units=latent_dimension, activation=activation)(encoded)

    return input_data, encoded

# DECODER
# input_dimension: size of original initial data
# encoded: output of the encoder
# activation: activation function.
# h1, h2, h3, h4, h5: integer, hidden layer dimensions. h3 and h4 are optional.
# num_layers: integer, number of layers. at least 3, at most 5.
# dropout: optional, boolean, if dropout to be used or not
def decoder(input_dimension, encoded, activation, h1, h2, h5, h3=80, h4=50, num_layers=4, dropout=False):
    # dense layer 1
    decoded = tf.keras.layers.Dense(units=h5, activation=activation)(encoded)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    if dropout:
        decoded = tf.keras.layers.Dropout(0.3)(decoded)
    if num_layers>4:
        # dense layer 2
        decoded = tf.keras.layers.Dense(units=h4, activation=activation)(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        if dropout:
            decoded = tf.keras.layers.Dropout(0.3)(decoded)
    if num_layers>3:
        # dense layer 3
        decoded = tf.keras.layers.Dense(units=h3, activation=activation)(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        if dropout:
            decoded = tf.keras.layers.Dropout(0.3)(decoded)
    # dense layer 4
    decoded = tf.keras.layers.Dense(units=h2, activation=activation)(decoded)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    if dropout:
        decoded = tf.keras.layers.Dropout(0.3)(decoded)
    # dense layer 5
    decoded = tf.keras.layers.Dense(units=h1, activation=activation)(decoded)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    if dropout:
        decoded = tf.keras.layers.Dropout(0.3)(decoded)
    # output layer
    decoded = tf.keras.layers.Dense(units=input_dimension, activation=tf.nn.sigmoid)(decoded)

    return decoded
    
    
# ENCODER CNN
# input_dimension: size of original initial data
# activation: activation function.
# latent_dimension: integer, dimension of latent space
def encoder_cnn(input_size, activation, latent_dimension):  
    input_data_cnn = tf.keras.Input(shape=(input_size))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, padding='same')(input_data_cnn)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation=activation, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    encod = tf.keras.layers.Dense(units=latent_dimension, activation=activation)(x)

    return input_data_cnn, encod, x.shape[1]   
    
# DECODER CNN
# encod: output of the encoder
# activation: activation function.
# x_shape: size of the flattened last layer of encoder
def decoder_cnn(encod, activation, x_shape):  
    dec = tf.keras.layers.Dense(x_shape, activation=activation)(encod)
    dec = tf.keras.layers.Reshape((3, 3, 128))(dec)
    dec = tf.keras.layers.Conv2D(128, (3, 3), activation=activation, padding='same')(dec)
    dec = tf.keras.layers.BatchNormalization()(dec)
    dec = tf.keras.layers.UpSampling2D((2, 2))(dec)
    dec = tf.keras.layers.Dropout(0.3)(dec)
    dec = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same')(dec)
    dec = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same')(dec)
    dec = tf.keras.layers.BatchNormalization()(dec)
    dec = tf.keras.layers.UpSampling2D((2, 2))(dec)
    dec = tf.keras.layers.Dropout(0.3)(dec)
    dec = tf.keras.layers.Conv2D(32, (3, 3), activation=activation)(dec)
    dec = tf.keras.layers.Conv2D(32, (3, 3), activation=activation, padding='same')(dec)
    dec = tf.keras.layers.BatchNormalization()(dec)
    dec = tf.keras.layers.UpSampling2D((2, 2))(dec)
    decod = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(dec)

    return decod
    