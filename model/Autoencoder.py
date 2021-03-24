import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


"""
The Autoencoder class is defined as an encoder-decoder data compression
alogirthm, whose functions are:
  1) data-specific (works best on reconstructing data similar to training data)
  2) lossy (reconstruction will be a bit degraded, similar to JPEG compression)
  3) learns automatically from data examples (new feature engineering not req.)

Source code attributed to Steve Shao, 'Contextual Topic Identification' (README)
"""


class Autoencoder:
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim # size of encoded representation
        self.activation = activation
        self.epochs = epochs # number of iterations
        self.batch_size = batch_size
        self.autoencoder = None # the model maps an input to its reconstruction
        self.encoder = None # encoded representation of the input
        self.decoder = None # lossy representaiton of the input
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,)) # encoded 32-dimensional input
        decoder_layer = self.autoencoder.layers[-1] # retrieve last layer of model
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, X):
        """
        preapre input data and train Autoencoder for 200 epochs
        """
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(X_train, X_train,
                                        epochs=200,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(X_test, X_test), verbose=0)
