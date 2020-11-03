import tables  # load compressed data files
import numpy as np
import pandas as pd
# from crf import CRF
from tf2crf import CRF
import tensorflow as tf
import keras.backend as K
#from crf_nlp_architect import CRF   ?????
import matplotlib.pyplot as plt
from evaluation import evaluation
from tensorflow import constant  # used to convert array/list to a Keras Tensor
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.models import Model, Input
from keras.callbacks import TensorBoard
from keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional



pd.set_option('display.max_columns', None)


# ======================================================================================================================
# Set data generators for batch training
# ======================================================================================================================

# Set batch size, and, the train and test data size
batch_size = 1024  # set during pre-processing (set in file preprocessing.py)
train_data_size = 4189874  # 530809  # 4189874 [ THE NUMBER OF TRAIN SENTENCES\DOCS ]  # the total size of train data
validation_data_size = 158451  # 20000  # 158451 [ THE NUMBER OF VALIDATION SENTENCES\DOCS ]  # the total size of test data
test_data_size = 157714  # 20000  # SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data


# Set INPUT and OUTPUT layer size
MAX_LEN = 70  # 500  # max length of abstract and title (together) (full text train set: 2881)
VECT_SIZE = 100  # the GloVe vector size
number_labels = 2  # 2 labels, keyword (KP) and Non-keyword (Non-KP)

print(MAX_LEN)
print(VECT_SIZE)


# ======================================================================================================================
# Define train/test data file names
# ======================================================================================================================
'''
# [FULL ABSTRACT TEXT - train_data_size = 530809] Define the file paths and names for TRAINING data
x_train_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed.hdf'
y_train_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed.hdf'

# [FULL ABSTRACT TEXT - validation_data_size = 20000] Define the file paths and names for VALIDATION data to tune model parameters
x_validate_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed.hdf'
y_validate_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed.hdf'

'''
# [SENTENCES ABSTRACT TEXT - train_data_size = 4189874] Define the file paths and names for TRAINING data
x_train_filename = 'data\\preprocessed_data\\x_TRAIN_SENTENC_data_preprocessed.hdf'
y_train_filename = 'data\\preprocessed_data\\y_TRAIN_SENTENC_data_preprocessed.hdf'

# [SENTENCES ABSTRACT TEXT - validation_data_size = 158451] Define the file paths and names for VALIDATION data to tune model parameters
x_validate_filename = 'data\\preprocessed_data\\x_VALIDATION_SENTENC_data_preprocessed.hdf'
y_validate_filename = 'data\\preprocessed_data\\y_VALIDATION_SENTENC_data_preprocessed.hdf'



# ======================================================================================================================
# Define file names for TESTING-EVALUATION of the final model (GOLD sets, preprocessed document text + keyphrases)
# ======================================================================================================================

# Full abstract
'''
# [ test_data_size = 20000 ]
x_test_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed.hdf'  # kp20k
x_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT'  # kp20k
y_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT'  # kp20k
'''
'''
# [ test_data_size = 210 ]
x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_preprocessed_TEXT'
'''
'''
# [ test_data_size = 2303 ]
x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_TEST_vectors.hdf'
x_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_preprocessed_TEXT'
'''

# Abstract in sentences

# [ test_data_size = 157714 ]
x_test_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed.hdf'  # kp20k
x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_preprocessed_TEXT'  # kp20k
y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_preprocessed_TEXT'  # kp20k

'''
# [ test_data_size = 1683 ]
x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
'''
'''
# [ test_data_size = 17605 ]
x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
'''

# Fulltext in sentences
'''
# [ test_data_size = 77198 ]
x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
'''
'''
# [ test_data_size = 787595 ]
x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_TEST_data_preprocessed.hdf'
x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
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
        #print(x)
        print('X SHAPE AFTER', np.array(x, dtype=object).shape)

    if not y_filename == '':    # for TEST data read only the x values
        # Read y batches for testing from file (pre-processed)
        with tables.File(y_filename, 'r') as h5f:
            y = h5f.get_node('/y_data' + str(batch_number)).read()  # get a specific chunk of data
            #print(y)
            print('y SHAPE AFTER', np.array(y, dtype=object).shape)

    '''
    print(y)
    here = [1 if doc[:, 1].any() else 0 for doc in y]
    print('y SHAPE AFTER', np.array(y, dtype=object).shape)
    if any(here):
        print('THERE ARE KEYPHRASES')
    else:
        print('THERE ARE NOOOOOOT KEYPHRASES')
    '''

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

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
# ======================================================================================================================
# Set data generators for batch training/validation/prediction
# ======================================================================================================================

# ceil of the scalar x is THE SMALLER INTEGER i, such that i >= x
steps_per_epoch = np.ceil(train_data_size/batch_size)  # number of training batches
validation_steps = np.ceil(validation_data_size/batch_size)  # number of validation and testing batches (same size)
test_steps = np.ceil(test_data_size/batch_size)  # number of validation and testing batches (same size)
print('steps_per_epoch', steps_per_epoch)
print('validation_steps', validation_steps)
print('test_steps', test_steps)

# (number of batches) -1, because batches start from 0
training_batch_generator = batch_generator(x_train_filename, y_train_filename, batch_size, steps_per_epoch - 1)  # training batch generator
validation_batch_generator = batch_generator(x_validate_filename, y_validate_filename, batch_size, validation_steps - 1)  # validation batch generator
test_batch_generator = batch_generator(x_test_filename, '', batch_size, test_steps - 1)  # testing batch generator


# ======================================================================================================================
# Bi-LSTM-CRF
# ======================================================================================================================

# Model definition
inpt = Input(shape=(MAX_LEN, VECT_SIZE))
# input_dim: Size of the vocabulary, i.e. maximum integer index + 1
# output_dim: Dimension of the dense embedding
# input_shape: 2D tensor with shape (batch_size, input_length)
'''
model = Embedding(input_dim=n_words+1, output_dim=100,  # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN, mask_zero=True)(inpt)  # default: 20-dim embedding
'''

# n_words: vocabulary - number of words - of the dataset
model = Embedding(TXT_VOCAB+1, output_dim=100, input_length=MAX_LEN,
                  # weights=[X_train],  # use GloVe vectors as initial weights
                  trainable=True)(inpt)  # name='word_embedding', mask_zero=True


# recurrent_dropout=0.1 (recurrent_dropout: 10% possibility to drop of the connections that simulate LSTM memory cells)
# units = 100 / 0.55 = 182 neurons (to account for 0.55 dropout)
model = Bidirectional(LSTM(units=100, return_sequences=True))(model)  # input_shape=(1, MAX_LEN, VECT_SIZE)
# model = Dropout(0.3)(model)
# model = TimeDistributed(Dense(number_labels, activation="relu"))(model)  # a dense layer as suggested by neuralNer
model = Dense(number_labels, activation='linear')(model)  # activation=None (they are the same)
crf = CRF(number_labels)  # CRF layer { SHOULD I SET -> number_labels+1 (+1 -> PAD) }
out = crf(model)  # output
model = Model(inputs=inpt, outputs=out)


# set learning rate
#lr_rate = InverseTimeDecay(initial_learning_rate=0.05, decay_rate=4, decay_steps=steps_per_epoch)
# lr_rate = ExponentialDecay(initial_learning_rate=0.01, decay_rate=0.5, decay_steps=10000)

# set optimizer
# decay=learning_rate / epochs
# CASE 1: decay=0.01
# CASE 2: decay=0.1/5
opt = SGD(learning_rate=0.1, decay=0.01, momentum=0.9, clipvalue=3.0, clipnorm=2.0)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]
#opt = SGD(learning_rate=lr_rate, clipvalue=3.0, clipnorm=2.0, momentum=0.9)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]
'''
2 2 
val_loss: 54.1747         loss: 33.4753
Exact Match
Precision: 0.0206
Recall: 0.0002
F1-score: 0.0003
Partial Match
Precision: 0.4515
Recall: 0.0113
F1-score: 0.0220
'''

# compile Bi-LSTM-CRF
model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])
# model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.viterbi_accuracy])

print('BEFORE TRAINING', model.get_weights())


# ======================================================================================================================
# Model Training
# ======================================================================================================================

# Track and report learning rate
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        lr = K.eval(tf.cast(optimizer.lr, dtype=tf.float32) * (1. / (1. + tf.cast(optimizer.decay, dtype=tf.float32) * tf.cast(optimizer.iterations, dtype=tf.float32))))
        logs.update({'lr-SGD': lr})
        super().on_epoch_end(epoch, logs)

'''
from keras.callbacks import Callback

# Track and report learning rate
class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(tf.cast(optimizer.lr, dtype=tf.float32) * (1. / (1. + tf.cast(optimizer.decay, dtype=tf.float32) * tf.cast(optimizer.iterations, dtype=tf.float32))))
        print('\nlr: {:.6f}\n'.format(lr))
'''

my_callbacks = [
    # save the weights of the model with the best loss after each epoch
    tf.keras.callbacks.ModelCheckpoint(filepath='pretrained_models\\checkpoint\\model.{epoch:02d}-{val_loss:.2f}.h5',
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True),
    LRTensorBoard(log_dir="/tmp/tb_log")  # report learning rate after each epoch
]

# tf.keras.callbacks.TensorBoard(log_dir='./logs')
# tensorboard --logdir logs/validation         [ USE ON TERMINAL ]


# Train model
history = model.fit(x=training_batch_generator, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_batch_generator, validation_steps=validation_steps - 1,  # -1, because it reads more times than actual needed
                    epochs=5, callbacks=my_callbacks, verbose=2)  # epochs=5

# [MANDATORY] Convert data to either a Tensorflow tensor (for CRF layer) or a numpy array
# y_train = constant(y_train)  # convert array/list to a Keras Tensor
# y_train = np.array(y_train)  # convert array/list to a numpy array

model.summary()

print('AFTER TRAINING', model.get_weights())


# ======================================================================================================================
# Track model loss per epoch
# ======================================================================================================================

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('model_loss_per_epoch.png')  # save the plot of model's loss per epoch to file


# ======================================================================================================================
# Predict on validation data
# ======================================================================================================================

print('\nPredicting...')
y_pred = model.predict(x=test_batch_generator, steps=test_steps)  # steps=validation_steps, because it does not read the last batch
print(y_pred)
print('\nY_PRED SHAPE', np.array(y_pred, dtype=object).shape)


# ======================================================================================================================
# Evaluation
# ======================================================================================================================

evaluation(y_pred, x_filename, y_filename)  # evaluate the model's performance


# ======================================================================================================================
# Model Saving
# ======================================================================================================================

# Save the model (weights)
# save_all_weights | load_all_weights: saves model and optimizer weights (save_weights and save)
model.save_weights("pretrained_models\\fulltext_model_weights.h5")  # sentences_model_weights.h5

'''
# `assert_consumed` can be used as validation that all variable values have been restored from the checkpoint. 
# See `tf.train.Checkpoint.restore` for other methods in the Status object.
print(load_status.assert_consumed())

# Check that all of the pretrained weights have been loaded.
for a, b in zip(pretrained.weights, model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())
'''


# Save the model (architecture, loss, metrics, optimizer state, weights)
model.save('pretrained_models\\fulltext_bi_lstm_crf_dense_linear.h5')  # sentences_bi_lstm_crf_dense_linear.h5
'''
# Load the model
from keras.models import load_model
model = load_model('pretrained_models\\fulltext_bi_lstm_crf_dense_linear.h5', custom_objects={'CRF': CRF(number_labels), 'num_classes': number_labels})  #  , 'loss': crf.loss, 'metrics': [crf.accuracy]
'''

# ======================================================================================================================
# Plot the layer architecture of LSTM
# ======================================================================================================================

plot_model(model, "schemas\\bi-lstm-crf_architecture.png", show_shapes=True)
