import tables  # load compressed data files
import numpy as np
import pandas as pd
import tensorflow as tf
from crf import CRF
# from tf2crf import CRF
from evaluation import evaluation
from tensorflow import constant  # used to convert array/list to a Keras Tensor
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.models import Model, Input
from keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

pd.set_option('display.max_columns', None)


# ======================================================================================================================
# Define file names for TESTING-EVALUATION of the final model (GOLD sets, preprocessed document text + keyphrases)
# ======================================================================================================================

# Full abstract
x_test_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed.hdf'  # kp20k
x_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT'  # kp20k
y_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT'  # kp20k
'''
x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_preprocessed_TEXT'
'''
'''
x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_TEST_vectors.hdf'
x_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_preprocessed_TEXT'
'''

# Abstract in sentences
'''
x_test_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed.hdf'  # kp20k
x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_preprocessed_TEXT'  # kp20k
y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_preprocessed_TEXT'  # kp20k
'''
'''
x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
'''
'''
x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
'''

# Fulltext
'''
x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
'''
'''
x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_TEST_data_preprocessed.hdf'
#x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
#y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
'''


# ======================================================================================================================
# Read train data
# ======================================================================================================================

def batch_generator(x_filename, y_filename, batch_size, number_of_batches):
    '''
    Continuously generates the data batches
    :param x_filename: the file name that contains pre-processed data of x
    :param y_filename: the file name that contains pre-processed data of y
    :param batch_size: the size of each batch
    :param number_of_batches: the total number of batches
    :return: yields data batches of x and y (Keras generator should not return and should continuously run)
    '''
    current_batch_number = 0  # the identifier used for each batch of data (ex. 0, 10000, 20000, 30000, etc.)

    if 'TRAIN' in x_filename:  # for training yield a batch as a dummy, because it does not read the first ever batch
        yield load_data(x_filename, y_filename, current_batch_number)  # needed to avoid loosing the first batch of data

    while True:
        yield load_data(x_filename, y_filename, current_batch_number)

        # calculate the identifier of each batch of data
        if current_batch_number < batch_size * number_of_batches:
            current_batch_number += batch_size
        else:
            current_batch_number = 0


def load_data(x_filename, y_filename, batch_number):
    '''
    Load the data batch-by-batch
    :param x_filename: the file name that contains pre-processed data of x
    :param y_filename: the file name that contains pre-processed data of y
    :param batch_number: the current number of batch for a specific iteration (ex. 2nd batch out of 10)
    :return: pre-processed data of x and y (y as tensor)
    '''

    print('batch_number', batch_number)
    print(x_filename)

    # Read X batches for testing from file (pre-processed)
    with tables.File(x_filename, 'r') as h5f:
        x = h5f.get_node('/x_data' + str(batch_number)).read()  # get a specific chunk of data
        # print(x)
        print('X SHAPE AFTER', np.array(x, dtype=object).shape)

    if not y_filename == '':  # for TEST data read only the x values
        # Read y batches for testing from file (pre-processed)
        with tables.File(y_filename, 'r') as h5f:
            y = h5f.get_node('/y_data' + str(batch_number)).read()  # get a specific chunk of data
            # print(y)
            print('y SHAPE AFTER', np.array(y, dtype=object).shape)

    '''
        if 'TRAIN' in x_filename:  # for training return class weights as well
            """
                # FULL ABSTRACT
                KP count:  3706267 
                NON-KP count:  421469323

                # SENTENCES
                KP count:  7862742 
                NON-KP count:  75909213
            """
            # The weight of label 0 (Non-KP) is 1 and the weight of class 1 (KP) is the number of occurences of class 0 (421469323 / 3706267 = 113.718)
            sample_weights = [[1 if label[0] else 113.718 for label in instance] for instance in y]  # shape (1024, 2881)
            print('class_weights', np.array(sample_weights, dtype=float).shape)
            return x, constant(y), np.array(sample_weights, dtype=float)  # (inputs, targets, sample_weights)
    '''

    if y_filename == '':  # for TEST data return only the x values
        return x

    return x, constant(y)


# USE weights for all data? (train, validation and test???)


# ======================================================================================================================
# Bi-LSTM-CRF
# ======================================================================================================================

number_labels = 2  # 2 labels, keyword (KP) and Non-keyword (Non-KP)
MAX_LEN = 500  # 70  # 2881  # max length of abstract and title (together)
VECT_SIZE = 100  # the GloVe vector size
print(MAX_LEN)
print(VECT_SIZE)

# Model definition
inpt = Input(shape=(MAX_LEN, VECT_SIZE))
# input_dim: Size of the vocabulary, i.e. maximum integer index + 1
# output_dim: Dimension of the dense embedding
# input_shape: 2D tensor with shape (batch_size, input_length)
# recurrent_dropout: 10% possibility to drop of the connections that simulate LSTM memory cells
model = Bidirectional(LSTM(units=100, return_sequences=True))(inpt)  # variational biLSTM
# model = Dropout(0.3)(model)
# model = TimeDistributed(Dense(number_labels, activation="relu"))(model)  # a dense layer as suggested by neuralNer
model = Dense(number_labels, activation='linear')(model)  # activation=None (they are the same)
crf = CRF(number_labels)  # CRF layer { SHOULD I SET -> number_labels+1 (+1 -> PAD) }
out = crf(model)  # output
model = Model(inputs=inpt, outputs=out)

# set learning rate
# lr_rate = InverseTimeDecay(initial_learning_rate=0.05, decay_rate=0.01, decay_steps=10000)
# lr_rate = ExponentialDecay(initial_learning_rate=0.1, decay_rate=0.5)  # decay_steps=10000,

# set optimizer
opt = SGD(learning_rate=0.01, decay=0.01, momentum=0.9, clipvalue=3.0, clipnorm=2.0)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]
#opt = SGD(learning_rate=lr_rate, clipnorm=2.0, momentum=0.9)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]
# opt = RMSprop(learning_rate=lr_rate, momentum=0.9)

# compile Bi-LSTM-CRF
model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])
# model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.viterbi_accuracy])

print('BEFORE TRAINING', model.get_weights())
# ======================================================================================================================
# Data Model Parameters
# ======================================================================================================================

batch_size = 1024  # set during pre-processing (set in file preprocessing.py)
validation_test_data_size = 20000  # the total size of test data

# ceil of the scalar x is THE SMALLER INTEGER i, such that i >= x
validation_test_steps = np.ceil(validation_test_data_size / batch_size)  # number of validation and testing batches (same size)
print('validation_steps', validation_test_steps)

# (number of batches) -1, because batches start from 0
test_batch_generator = batch_generator(x_test_filename, '', batch_size, validation_test_steps - 1)  # testing batch generator


# ======================================================================================================================
# Model Loading
# ======================================================================================================================

# save_all_weights | load_all_weights: saves model and optimizer weights (save_weights and save)
load_status = model.load_weights("pretrained_models\\fulltext_model_weights.h5")  # sentences_model_weights.h5

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
#print(load_status.assert_consumed())


model.summary()

print('AFTER LOADING', model.get_weights())
# ======================================================================================================================
# Predict on validation data
# ======================================================================================================================

print('\nPredicting...')
y_pred = model.predict(x=test_batch_generator, steps=validation_test_steps)  # steps=validation_steps, because it does not read the last batch
print(y_pred)
print('\nY_PRED SHAPE', np.array(y_pred, dtype=object).shape)


# ======================================================================================================================
# Evaluation
# ======================================================================================================================

evaluation(y_pred, x_filename, y_filename)  # evaluate the model's performance


# ======================================================================================================================
# Plot the layer architecture of LSTM
# ======================================================================================================================

plot_model(model, "schemas\\bi-lstm-crf_architecture.png", show_shapes=True)
