from typing import Iterable
from tensorflow import keras
from keras.losses import MeanSquaredError


def autoencoder(n_input: int,
                n_bottleneck: int,
                n_layers: Iterable[int]):
    '''create an autoencoder model with separate encoder, decoder, and complatee model for training
    input: dimensionality of input data
    bottleneck: dimensionality to reduce to
    layers: in-between layers

    Rules!
      - must be descending in size
      - no layer may be larger than the input (waste)
      - resonableness of step size (weird to have: (50, 30, 29, 28, 3))
    
    '''
    pass


    inputs = keras.layers.Input(shape=(n_input, )) # hanging comma forces tuple of length 1
    x = inputs
    for layer_size in [n_input] + n_layers:
        x = keras.layers.Dense(layer_size, activation='relu') # saved as x before to do recursive...
    bottleneck = keras.layers.Dense(n_bottleneck, activation='relu')(x)

    dec_inputs = keras.layers.Dense(n_layers[-1], activation='relu')(bottleneck)

#more here


if __name__ == '__main__':
    #enc, dec, full = autoencoder(50, 3, (40, 30, 20, 10))


    from duq_330.wk4 import read_wine_data

    df = read_wine_data.read()
    labels = df['quality']
    features = [col for col in df.columns if col != 'quality']

    encoder_model, training_model = autoencoder(n_input=11,
                                                n_bottleneck=2,
                                                n_layers=[8,6,4])
    
    #min max scaling
    for col in features:
        features[col] -= features[col].min()
        features[col] /= features[col].max()


    print('Before')
    print(encoder_model.predict(features))

    # autoencoder defined as fitting output data equal to the input data
    training_model.fit(features.values, features.values,
                       epochs=8,
                       batch_size=32,
                       shuffle=True)
    
    print('After Training')
    print(encoder_model.predict(features))

