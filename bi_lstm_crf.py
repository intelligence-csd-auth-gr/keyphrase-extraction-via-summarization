import tables
import numpy as np
import pandas as pd
from tf2crf import CRF
from tensorflow import constant  # used to convert array/list to a Keras Tensor
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.models import Model, Input
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
# Define train/test data file names
# ======================================================================================================================
'''
X = np.load('data\\preprocessed_data\\dummy_x_train_data_preprocessed.npy')
y = np.load('data\\preprocessed_data\\dummy_y_train_data_preprocessed.npy')
'''

# Define the file paths and names for TRAINING data
x_train_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed'
y_train_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed'
'''
# Define the file paths and names for SAMPLE OF TRAIN data
x_train_filename = 'data\\preprocessed_data\\dummy_x_train_data_preprocessed'
y_train_filename = 'data\\preprocessed_data\\dummy_y_train_data_preprocessed'
'''
# Define the file paths and names for VALIDATION data to tune model parameters
x_validate_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
y_validate_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'

# Define the file paths and names for TESTING data to evaluate the final model
x_test_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed'
y_test_filename = 'data\\preprocessed_data\\y_TEST_data_preprocessed'


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
    with tables.File(x_filename + '.hdf', 'r') as h5f:
        x = h5f.get_node('/x_data' + str(batch_number)).read()  # get a specific chunk of data
        #print(x)
        print('X SHAPE AFTER', np.array(x, dtype=object).shape)

    # Read y batches for testing from file (pre-processed)
    with tables.File(y_filename + '.hdf', 'r') as h5f:
        y = h5f.get_node('/y_data' + str(batch_number)).read()  # get a specific chunk of data
        #print(y)
        print('y SHAPE AFTER', np.array(y, dtype=object).shape)

    if 'TRAIN' in x_filename:  # for training return class weights as well
        # The weight of label 0 (Non-KP) is 1 and the weight of class 1 (KP) is the magnitude of times of that of class 0 ()
        sample_weights = [[1.0 if label[0] else 100000.0 for label in instance] for instance in y]  # shape (256, 2881)
        print('class_weights', np.array(sample_weights, dtype=float).shape)
        return x, constant(y), np.array(sample_weights, dtype=float)  # (inputs, targets, sample_weights)
    return x, constant(y)


# ======================================================================================================================
# Bi-LSTM-CRF
# ======================================================================================================================

number_labels = 2  # 2 labels, keyword (KP) and Non-keyword (Non-KP)
MAX_LEN = 2881  # max length of abstract and title (together)
VECT_SIZE = 100  # the GloVe vector size
print(MAX_LEN)
print(VECT_SIZE)
# Model definition
input = Input(shape=(MAX_LEN, VECT_SIZE))
# input_dim: Size of the vocabulary, i.e. maximum integer index + 1
# output_dim: Dimension of the dense embedding
# input_shape: 2D tensor with shape (batch_size, input_length)
'''
# n_words: Number of words in the dataset
model = Embedding(input_dim=n_words+1, output_dim=100,  # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
                  
                  
model = Embedding(TXT_VOCAB, output_dim=100, input_length=MAX_LEN,
                      weights=[X_train],  # use GloVe vectors as initial weights
                      name='word_embedding', trainable=True, mask_zero=True)(input)
'''


#from crf import CRF


#from crf_nlp_architect import CRF


model = Dropout(0.55)(input)
# recurrent_dropout: 10% possibility to drop of the connections that simulate LSTM memory cells
model = Bidirectional(LSTM(units=100 // 2, return_sequences=True,  # input_shape=(1, MAX_LEN, VECT_SIZE),
                           recurrent_dropout=0.1))(model)              # variational biLSTM
model = Dropout(0.55)(model)
#model = TimeDistributed(Dense(number_labels, activation="relu"))(model)  # a dense layer as suggested by neuralNer
model = Dense(number_labels, activation='linear')(model)  # activation=None (they are the same)
crf = CRF(number_labels)  # CRF layer, number_labels+1 (+1 -> PAD)
out = crf(model)  # output
model = Model(inputs=input, outputs=out)


# set learning rate
lr_rate = ExponentialDecay(initial_learning_rate=0.015, decay_steps=10000, decay_rate=0.05)
# set optimizer
opt = RMSprop(learning_rate=lr_rate, momentum=0.9)
# compile Bi-LSTM-CRF
model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])
#model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.viterbi_accuracy])


# ======================================================================================================================
# Model Training
# ======================================================================================================================

# 32 64
batch_size = 256  # set during pre-processing (set in file preprocessing.py)
train_data_size = 5120#530809  # the total size of train data
validation_test_data_size = 20000  # the total size of test data

# ceil of the scalar x is THE SMALLER INTEGER i, such that i >= x
steps_per_epoch = np.ceil(train_data_size/batch_size)  # number of training batches
validation_test_steps = np.ceil(validation_test_data_size/batch_size)  # number of validation and testing batches (same size)
print('steps_per_epoch', steps_per_epoch)
print('validation_steps', validation_test_steps)

# (number of batches) -1, because batches start from 0
training_batch_generator = batch_generator(x_train_filename, y_train_filename, batch_size, steps_per_epoch - 1)  # training batch generator
validation_batch_generator = batch_generator(x_validate_filename, y_validate_filename, batch_size, validation_test_steps - 1)  # validation batch generator
test_batch_generator = batch_generator(x_test_filename, y_test_filename, batch_size, validation_test_steps - 1)  # testing batch generator

# Train model
model.fit(x=training_batch_generator, steps_per_epoch=steps_per_epoch,
          validation_data=validation_batch_generator, validation_steps=validation_test_steps - 1,  # -1, because it reads more times than actual needed
          epochs=2, verbose=2)  # epochs=200

# [MANDATORY] Convert data to either a Tensorflow tensor (for CRF layer) or a numpy array
# y_train = constant(y_train)  # convert array/list to a Keras Tensor
# y_train = np.array(y_train)  # convert array/list to a numpy array

model.summary()


# ======================================================================================================================
# Predict on validation data
# ======================================================================================================================

print('\nPredicting...')
#y_pred = model.predict(X_test)
y_pred = model.predict(x=test_batch_generator, steps=validation_test_steps)  # steps=validation_steps, because it does not read the last batch
print('\nY_PRED SHAPE', np.array(y_pred, dtype=object).shape)


# ======================================================================================================================
# Load all validation target data (y_test\labels) data on memory (needed for evaluation)
# ======================================================================================================================

def load_y_test(y_test_filename, batch_size, number_of_batches):
    '''
    Load y_test for validation
    :param y_test_filename: the file name that contains pre-processed data of y_test
    :param batch_size: the size of each batch
    :param number_of_batches: the total number of batches
    :return: return y_test (y_test_flat) for validation
    '''

    y_test_batches = []  # save y_test for validation
    current_batch_number = 0  # the identifier used for each batch of data (ex. 0, 10000, 20000, 30000, etc.)
    while True:
        # Read X batches for testing from file (pre-processed)
        with tables.File(y_test_filename + '.hdf', 'r') as h5f:
            y_test_batches.append(h5f.get_node('/y_data' + str(current_batch_number)).read())  # get a specific chunk of data

        # calculate the identifier of each batch of data
        if current_batch_number < batch_size * number_of_batches:
            current_batch_number += batch_size
        else:
            y_test_flat = [y_label for y_batch in y_test_batches for y_label in y_batch]  # flatten the y_test (20000, 2881, 2)
            print('y_test SHAPE AFTER', np.array(y_test_flat, dtype=object).shape)
            return y_test_flat


y_test = load_y_test(y_test_filename, batch_size, validation_test_steps - 1)  # load y_test in memory


# ======================================================================================================================
# Convert y_test and y_pred from categorical (two columns, 1 for each label) to a single value label (1 column)
# ======================================================================================================================

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


# print('BEFORE y_pred', y_pred)
y_pred = pred2label(y_pred)  # convert y_test from categorical (two columns, 1 for each label) to a single value label
# print('AFTER y_pred', y_pred)

#print('BEFORE y_test', y_test)
y_test = pred2label(y_test)  # convert y_test from categorical (two columns, 1 for each label) to a single value label
#print('AFTER y_test', y_test)


# ======================================================================================================================
# Model Evaluation
# ======================================================================================================================

# pos_label: the label that the score is reported for (KP - keyphrase label is selected as it is more important)
print("F1-score for label KP: {:.2%}".format(f1_score(y_test, y_pred, pos_label='KP')))
print(classification_report(y_test, y_pred))


# ======================================================================================================================
# Model Saving
# ======================================================================================================================

# Save the model
model.save('bi_lstm_crf_dense_linear.h5')

# Load the model
from keras.models import load_model
model = load_model('bi_lstm_crf_dense_linear.h5')


# ======================================================================================================================
# Plot the layer architecture of LSTM
# ======================================================================================================================

plot_model(model, "bi-lstm-crf_architecture.png", show_shapes=True)
