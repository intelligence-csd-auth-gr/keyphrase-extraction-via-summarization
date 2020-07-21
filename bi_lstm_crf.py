import numpy as np
import pandas as pd
from tensorflow import constant  # used to convert array/list to a Keras Tensor
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.models import Model, Input
from sklearn.model_selection import train_test_split
from keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import f1_score, classification_report
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


#from keras_contrib.layers import CRF
#from keras_contrib.losses import crf_loss
#from keras_contrib.metrics import crf_accuracy
# model.compile(optimizer=opt, loss=crf.loss_function, metrics=[crf.accuracy])
#model.compile(optimizer=opt, loss=crf_loss, metrics=[crf_accuracy])


# pip -q install git+https://www.github.com/keras-team/keras-contrib.git sklearn-crfsuite


pd.set_option('display.max_columns', None)


# ======================================================================================================================
# Read data file
# ======================================================================================================================

X = np.load('data\preprocessed_data\dummy_x_train_data_preprocessed.npy')
y = np.load('data\preprocessed_data\dummy_y_train_data_preprocessed.npy')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# print(X_train.shape, X_test.shape, np.array(y_train).shape, np.array(y_test).shape)






# ======================================================================================================================
# CRF
# ======================================================================================================================

import tensorflow as tf
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood

def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        _, potentials, sequence_length, chain_kernel = y_pred

        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        return tf.reduce_mean(crf_loss), sum(self.losses)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"crf_loss": crf_loss, "internal_losses": internal_losses}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_losses = self.compute_loss(x, y, sample_weight)
        return {"crf_loss": crf_loss, "internal_losses": internal_losses}






# ======================================================================================================================
# Bi-LSTM-CRF
# ======================================================================================================================

number_labels = 2  # 2 labels, keyword (KP) and Non-keyword (Non-KP)

MAX_LEN = X_train.shape[1]  # shape: [number of examples | (max) length of examples with padding | size of vectors]
VECT_SIZE = X_train.shape[2]
print(MAX_LEN)
print(VECT_SIZE)
# Model definition
input = Input(shape=(MAX_LEN, VECT_SIZE))
# input_dim: Size of the vocabulary, i.e. maximum integer index + 1
# output_dim: Dimension of the dense embedding
# input_shape: 2D tensor with shape (batch_size, input_length)
'''
n_words: Number of words in the dataset
'''
'''
model = Embedding(input_dim=n_words+1, output_dim=100,  # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
                  
                  
model = Embedding(TXT_VOCAB, output_dim=100, input_length=MAX_LEN,
                      weights=[X_train],  # use GloVe vectors as initial weights
                      name='word_embedding', trainable=True, mask_zero=True)(input)
'''


#from crf import CRF


#from tf2crf import CRF


#from crf_nlp_architect import CRF

'''
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

model.compile(optimizer=opt, loss=crf_loss, metrics=[crf_accuracy])
'''

model = Dropout(0.55)(input)
# recurrent_dropout: 10% possibility to drop of the connections that simulate LSTM memory cells
model = Bidirectional(LSTM(units=100 // 2, return_sequences=True,  # input_shape=(1, MAX_LEN, VECT_SIZE),
                           recurrent_dropout=0.1))(model)              # variational biLSTM
model = Dropout(0.55)(model)
#model = TimeDistributed(Dense(100, activation="relu"))(model)  # a dense layer as suggested by neuralNer
#model = Dense(number_labels, activation=None)(model)
crf = CRF(number_labels)  # CRF layer, number_labels+1 (+1 -> PAD)
out = crf(model)  # output
model = Model(inputs=input, outputs=out)
model = ModelWithCRFLoss(model)


# set learning rate
lr_rate = ExponentialDecay(initial_learning_rate=0.015, decay_steps=10000, decay_rate=0.05)
# set optimizer
opt = RMSprop(learning_rate=lr_rate, momentum=0.9)
# compile Bi-LSTM-CRF
model.compile(optimizer=opt)
#model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])
#model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.viterbi_accuracy])


# ======================================================================================================================
# Model Training
# ======================================================================================================================

# [MANDATORY] Convert data to either a Tensorflow tensor or a numpy array
y_train = constant(y_train)  # convert array/list to a Keras Tensor
# y_train = np.array(y_train)  # convert array/list to a numpy array

# Train model
history = model.fit(X_train,  y_train, batch_size=200, validation_split=0.1, epochs=1, verbose=2)  # batch_size=10, epochs=200,

model.summary()

# ======================================================================================================================
# Model Evaluation
# ======================================================================================================================

y_pred = model.predict(X_test)


def pred2label(all_abstract_preds_preds):
    '''
    Converts prediction set and test/validation set from two columns (one for each label value)
    to just one column with the number of the corresponding label
    [ initial array: [1, 0] => final array: [0] ]   -   [ initial array: [0, 1] => final array: [1] ]
    :param all_abstract_preds_preds: array with predictions or test/validation set [documents/abstracts, number of words]
    :return: flattened array that contains the prediction for each word [number of total words of all abstracts]
    '''
    preds = []
    for abstract_preds in all_abstract_preds_preds:
        for word_pred in abstract_preds:
            # the position of the max value is corresponding to the actual label value (0: Non-KP, 1: KP)
            preds.append('KP' if np.argmax(word_pred) else 'Non-KP')
    return preds


print('BEFORE y_pred', y_pred)
y_pred = pred2label(y_pred)
print('AFTER y_pred', y_pred)

print('BEFORE y_test', y_test)
y_test = pred2label(y_test)
print('AFTER y_test', y_test)

# Evaluation metrics
# pos_label: the label that the score is reported for (KP - keyphrase label is selected as it is more important)
print("F1-score for label KP: {:.2%}".format(f1_score(y_test, y_pred, pos_label='KP')))
print(classification_report(y_test, y_pred))


# ======================================================================================================================
# Plot the layer architecture of LSTM
# ======================================================================================================================

plot_model(model, "bi-lstm-crf_architecture.png", show_shapes=True)
