import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
# from keras.models import load_model


class Attention(tf.keras.Model):
	def __init__(self, units):
		super(Attention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, features, hidden):
		# hidden shape == (batch_size, hidden size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden size)
		# we are doing this to perform addition to calculate the score
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying score to self.V
		# the shape of the tensor before applying self.V is (batch_size, max_length, units)
		score = tf.nn.tanh(
			self.W1(features) + self.W2(hidden_with_time_axis))
		# attention_weights shape == (batch_size, max_length, 1)
		attention_weights = tf.nn.softmax(self.V(score), axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights


def bidirectionalLSTM(dataset, pretrained_weights = None):
    if pretrained_weights:
        fname = 'models/' + pretrained_weights
        model = load_model(fname)
        return model

    n_timesteps = dataset.shape[1]
    n_block = dataset.shape[2]

    # in_tensor = tf.keras.layers.Input((n_timesteps,n_block,),name='Input')
    # bidir = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20))(in_tensor)
    # out = tf.keras.layers.Dense(2, activation='softmax')(bidir) #was 3

    in_tensor = tf.keras.layers.Input((n_timesteps,n_block,),name='Input')
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True), name="bi_lstm_0")(in_tensor)
    (lstm, forward_h, forward_c, backward_h, backward_c) = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True, return_state=True ,name='bi_lstm_1'))(lstm)
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    context_vector, attention_weights = Attention(32)(lstm, state_h)
    dense1 = tf.keras.layers.Dense(20, activation="relu")(context_vector)
    dropout = tf.keras.layers.Dropout(0.05)(dense1)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

    # out = tf.keras.layers.Dense(2, activation='softmax')(bidir) #was 3

    model = tf.keras.models.Model( in_tensor,  out)


    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    if pretrained_weights is not None:
        fname = 'models/' + pretrained_weights
        self.load(fname)

    # print(model.summary())
    return model


def CNN_mod(dataset, pretrained_weights = None):
    if pretrained_weights:
        fname = 'models/' + pretrained_weights
        model = load_model(fname)
        return model

    n_timesteps = dataset.shape[1]
    n_block = dataset.shape[2]

    # in_tensor = tf.keras.layers.Input((n_timesteps,n_block,),name='Input')
    # bidir = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20))(in_tensor)
    # out = tf.keras.layers.Dense(2, activation='softmax')(bidir) #was 3

    in_tensor = tf.keras.layers.Input((n_timesteps,n_block,),name='Input')
    cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(in_tensor)
    cnn = tf.keras.layers.Conv1D(filters=128, kernel_size=10, activation='relu')(cnn)
    drop = tf.keras.layer.Dropout(0.5)(cnn)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(drop)


    # out = tf.keras.layers.Dense(2, activation='softmax')(bidir) #was 3

    model = tf.keras.models.Model( in_tensor,  out)


    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

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
