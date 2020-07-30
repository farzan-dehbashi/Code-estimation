import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
# from keras.models import load_model


def bidirectionalLSTM(dataset, pretrained_weights = None):
    if pretrained_weights:
        fname = 'models/' + pretrained_weights
        model = load_model(fname)
        return model

    n_timesteps = dataset.shape[1]
    n_block = dataset.shape[2]

    in_tensor = tf.keras.layers.Input((n_timesteps,n_block,),name='Input')
    bidir = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20))(in_tensor)
    out = tf.keras.layers.Dense(2, activation='softmax')(bidir) #was 3

    model = tf.keras.models.Model( in_tensor,  out)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    if pretrained_weights is not None:
        fname = 'models/' + pretrained_weights
        self.load(fname)

    # print(model.summary())
    return model


class BidirectionalLSTM(Model):

    def __init__(self, dataset, weightsFileName=None):
        super(BidirectionalLSTM, self).__init__()

        n_timesteps = dataset.shape[2]

        self.input = Input((n_timesteps,),name='Input')
        bidir = Bidirectional(LSTM(20, return_sequences=True))(self.input)
        out = Dense(3, activation='softmax')(bidir)

        super().__init__(inputs = self.input, outputs = out)

        print(self.summary())

        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
