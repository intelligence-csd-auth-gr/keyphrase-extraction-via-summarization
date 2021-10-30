import time
import sys
import tables  # load compressed data files
import numpy as np
import pandas as pd
# from crf import CRF
from tf2crf import CRF
from numpy import load
import tensorflow as tf
import sequence_evaluation
import traditional_evaluation
from datetime import timedelta
from tensorflow import constant  # used to convert array/list to a Keras Tensor
from argparse import ArgumentParser
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model, Input
from data_generator import DataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

pd.set_option('display.max_columns', None)

# count the running time of the program
start_time = time.time()

import os
# disable tensorflow GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ======================================================================================================================
# Argument parsing
# ======================================================================================================================

parser = ArgumentParser()

parser.add_argument("-sts", "--select_test_set", type=str,
                    help="select the test set to evaluate the model (options are:"
                         "'kp20k_full_abstract'"
                         "'nus_full_abstract'"
                         "'acm_full_abstract'"
                         "'semeval_full_abstract'"
                         ""
                         "'kp20k_sentences_abstract'"
                         "'nus_sentences_abstract'"
                         "'acm_sentences_abstract'"
                         "'semeval_sentences_abstract'"
                         ""
                         "'nus_sentences_fulltext'"
                         "'acm_sentences_fulltext'"
                         "'semeval_sentences_fulltext'"
                         ""
                         "'nus_paragraph_fulltext'"
                         "'acm_paragraph_fulltext'"
                         "'semeval_paragraph_fulltext'"
                         ""
                         "'nus_220_first_3_paragraphs'"
                         "'acm_220_first_3_paragraphs'"
                         "'semeval_220_first_3_paragraphs'"
                         "'nus_400_first_3_paragraphs'"
                         "'acm_400_first_3_paragraphs'"
                         "'semeval_400_first_3_paragraphs'"
                         ""
                         "'nus_summarization'"
                         "'acm_summarization'"
                         "'semeval_summarization'"
                         ")"
                    )

parser.add_argument("-pmp", "--pretrained_model_path", type=str,
                    help="the path and the name of the pretrained model")

parser.add_argument("-sm", "--sentence_model", type=int, default=0,
                    help="choose which data to load (options are: True for sentence model or False for whole title and abstracts model)")

args = parser.parse_args()


# ======================================================================================================================
# Set data generators for batch training
# ======================================================================================================================

if args.sentence_model:
    # Set batch size, train and test data size
    batch_size = 256#224  # 1024  # set during pre-processing (set in file preprocessing.py)
    train_data_size = 4136306#4139868  # 530809  [ THE NUMBER OF TRAIN SENTENCES\DOCS ]  # the total size of train data
    validation_data_size = 156519#156836  # 20000  [ THE NUMBER OF VALIDATION SENTENCES\DOCS ]  # the total size of test data
    #test_data_size = 155801#156085  # 156085  SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data

    # Set INPUT layer size
    MAX_LEN = 40  # 70  # max length of abstract and title (together) (full text train set: 2763)
else:
    # Set batch size,train and test data size
    batch_size = 64  # 1024  # set during pre-processing (set in file preprocessing.py)
    train_data_size = 530390 #530809  # 4147964  # 530809  [ THE NUMBER OF TRAIN SENTENCES\DOCS ]  # the total size of train data
    validation_data_size = 20000  # 156836  # 20000  [ THE NUMBER OF VALIDATION SENTENCES\DOCS ]  # the total size of test data
    #test_data_size = 20000  # 20000  SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data

    # Set INPUT layer size
    # set to 220 for full-text split in paragraphs (paragraphs that have above 220 length exist - need to be cropped for sequence evaluation)
    MAX_LEN = 400  # 220   400   500   # max length of abstract and title (together) (full text train set: 2763)

# Set embedding size, OUTPUT layer size
VECT_SIZE = 100  # the GloVe vector size
number_labels = 2  # 2 labels, keyword (KP) and Non-keyword (Non-KP)

doc_vocab = 321352#321490 #298383  # 344858    #323895   # 100003  # the vocabulary of the train dataset

print('MAX_LEN of text', MAX_LEN)
print('VECT_SIZE', VECT_SIZE)
print('VOCABULARY', doc_vocab)


# ======================================================================================================================
# Define file names for TESTING-EVALUATION of the final model (GOLD sets, preprocessed document text + keyphrases)
# ======================================================================================================================

# Full abstract
if args.select_test_set=="kp20k_full_abstract":
    # [ test_data_size = 20000 ]
    test_data_size = 20000
    x_test_filename = 'data\\preprocessed_data\\data_train1\\x_TEST_data_preprocessed.hdf'  # kp20k
    y_test_filename = 'data\\preprocessed_data\\data_train1\\y_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\data_train1\\x_TEST_preprocessed_TEXT'  # kp20k
    y_filename = 'data\\preprocessed_data\\data_train1\\y_TEST_preprocessed_TEXT'  # kp20k
elif args.select_test_set=="nus_full_abstract":
    # [ test_data_size = 211 ]
    test_data_size = 211
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_preprocessed_TEXT'
elif args.select_test_set=="acm_full_abstract":
    # [ test_data_size = 2304 ]
    test_data_size = 2304
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_TEST_vectors.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_TEST_vectors'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_preprocessed_TEXT'
elif args.select_test_set=="semeval_full_abstract":
    # [ test_data_size = 244 ]
    test_data_size = 244
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_SEMEVAL_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_SEMEVAL_FULL_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_SEMEVAL_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_SEMEVAL_FULL_ABSTRACT_preprocessed_TEXT'

# Sentences abstract
elif args.select_test_set=="kp20k_sentences_abstract":
    # [ test_data_size = 155801 ]
    test_data_size = 155801
    x_test_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed.hdf'  # kp20k
    y_test_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_preprocessed_TEXT'  # kp20k
    y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_preprocessed_TEXT'  # kp20k
elif args.select_test_set=="nus_sentences_abstract":
    # [ test_data_size = 1673 ]
    test_data_size = 1673
    x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_abstract\\y_NUS_SENTEC_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
elif args.select_test_set=="acm_sentences_abstract":
    # [ test_data_size = 17481 ]
    test_data_size = 17481
    x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_abstract\\y_ACM_SENTENC_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
elif args.select_test_set=="semeval_sentences_abstract":
    # [ test_data_size = 1979 ]
    test_data_size = 1979
    x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_SEMEVAL_SENTEC_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_abstract\\y_SEMEVAL_SENTEC_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_SEMEVAL_SENTEC_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_SEMEVAL_SENTEC_ABSTRACT_preprocessed_TEXT'

# Sentences fulltext
elif args.select_test_set=="nus_sentences_fulltext":
    # [ test_data_size = 74219 ]
    test_data_size = 74219
    x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_NUS_SENTEC_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="acm_sentences_fulltext":
    # [ test_data_size = 770263 ]
    test_data_size = 770263
    x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_ACM_SENTENC_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="semeval_sentences_fulltext":
    # [ test_data_size = 75726 ]
    test_data_size = 75726
    x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_SEMEVAL_SENTEC_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_SEMEVAL_SENTEC_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_SEMEVAL_SENTEC_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_SEMEVAL_SENTEC_FULLTEXT_preprocessed_TEXT'

# Paragraphs fulltext
elif args.select_test_set=="nus_paragraph_fulltext":
    # [ test_data_size = 4744 ]
    test_data_size = 4744
    x_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_NUS_PARAGRAPH_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_NUS_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_NUS_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_NUS_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="acm_paragraph_fulltext":
    # [ test_data_size = 53083 ]
    test_data_size = 53083
    x_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_ACM_PARAGRAPH_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_ACM_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_ACM_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_ACM_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="semeval_paragraph_fulltext":
    # [ test_data_size = 5171 ]
    test_data_size = 5171
    x_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_SEMEVAL_PARAGRAPH_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_SEMEVAL_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_SEMEVAL_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_SEMEVAL_PARAGRAPH_FULLTEXT_preprocessed_TEXT'

# First 3 paragraphs
elif args.select_test_set=="nus_220_first_3_paragraphs":
    # [ test_data_size = 633 - LEN 220 ]
    MAX_LEN = 220
    test_data_size = 633
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="nus_400_first_3_paragraphs":
    # [ test_data_size = 633 - LEN 400 ]
    test_data_size = 633
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="acm_220_first_3_paragraphs":
    # [ test_data_size = 6910 - LEN 220 ]
    MAX_LEN = 220
    test_data_size = 6910
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="acm_400_first_3_paragraphs":
    # [ test_data_size = 6910 - LEN 400 ]
    test_data_size = 6910
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="semeval_220_first_3_paragraphs":
    # [ test_data_size = 732 - LEN 220 ]
    MAX_LEN = 220
    test_data_size = 732
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
elif args.select_test_set=="semeval_400_first_3_paragraphs":
    # [ test_data_size = 732 - LEN 400 ]
    test_data_size = 732
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'

# Summarization of abstract and fulltext
elif args.select_test_set=="nus_summarization":
    # [ test_data_size = 211 ]
    test_data_size = 211
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_NUS_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_NUS_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\summarization_experiment\\x_NUS_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_NUS_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
elif args.select_test_set=="acm_summarization":
    # [ test_data_size = 2304 ]
    test_data_size = 2304
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_ACM_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_ACM_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\summarization_experiment\\x_ACM_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_ACM_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
elif args.select_test_set=="semeval_summarization":
    # [ test_data_size = 244 ]
    test_data_size = 244
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_SEMEVAL_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_SEMEVAL_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\summarization_experiment\\x_SEMEVAL_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_SEMEVAL_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
else:
    print('WRONG ARGUMENTS! - please fill the argument "-sts" or "--select_test_set" with one of the proper values')
    sys.exit()


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
            KP count:  3887080 (78315988 / 3887080 = 20.14)
            KP WORDS count:  6590980 (78315988 / 6590980 = 11.88)
            NON-KP count:  78315988

            # SENTENCES
            KP count:  3863958  (77728129 / 3863958 = 20.11)
            KP WORDS count:  6534606  (77728129 / 6534606 = 11.89)
            NON-KP TEST count:  77728129
        """
        # The weight of label 0 (Non-KP) is 1 and the weight of class 1 (KP) is the number of occurences of class 0 (78315988 / 6590980 = 11.88)
        sample_weights = [[1 if label[0] else 11.88 for label in instance] for instance in y]
        print('class_weights', np.array(sample_weights, dtype=float).shape)
        return x, constant(y), np.array(sample_weights, dtype=float)  # (inputs, targets, sample_weights)
    '''

    if y_filename == '':  # for TEST data return only the x values
        return x

    return x, constant(y)


# USE weights for all data? (train, validation and test???)


# ======================================================================================================================
# Load GloVe embedding
# ======================================================================================================================

# load dict of arrays
dict_data = load('data\\embedding_matrix.npz')
# extract the first array
embedding_matrix = dict_data['arr_0']
# print the array
print(embedding_matrix)


# ======================================================================================================================
# Bi-LSTM-CRF
# ======================================================================================================================

from keras.regularizers import l1
from keras.constraints import max_norm
# Model definition
inpt = Input(shape=(None,))  # MAX_LEN, VECT_SIZE
# input_dim: Size of the vocabulary, i.e. maximum integer index + 1
# output_dim: Dimension of the dense embedding
# input_shape: 2D tensor with shape (batch_size, input_length)

# doc_vocab: vocabulary - number of words - of the train dataset
model = Embedding(doc_vocab, output_dim=100, input_length=None,  # n_words + 2 (PAD & UNK)
                  weights=[embedding_matrix],  # use GloVe vectors as initial weights
                  mask_zero=True, trainable=True, activity_regularizer=l1(0.0000001))(inpt)  # name='word_embedding'

# recurrent_dropout=0.1 (recurrent_dropout: 10% possibility to drop of the connections that simulate LSTM memory cells)
# units = 100 / 0.55 = 182 neurons (to account for 0.55 dropout)
model = Bidirectional(LSTM(units=100, return_sequences=True, activity_regularizer=l1(0.0000000001), recurrent_constraint=max_norm(2)))(model)  # input_shape=(1, MAX_LEN, VECT_SIZE)
# model = Dropout(0.3)(model)  # 0.5
# model = TimeDistributed(Dense(number_labels, activation="relu"))(model)  # a dense layer as suggested by neuralNer
model = Dense(number_labels, activation=None)(model)  # activation='linear' (they are the same)
crf = CRF()  # CRF layer { SHOULD I SET -> number_labels+1 (+1 -> PAD) }
out = crf(model)  # output
model = Model(inputs=inpt, outputs=out)

# set optimizer
# decay=learning_rate / epochs
opt = SGD(learning_rate=0.0, momentum=0.9, clipvalue=5.0)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]
#opt = SGD(learning_rate=0.05, decay=0.01, momentum=0.9, clipvalue=5.0)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]

# compile Bi-LSTM-CRF
model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])
# model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.viterbi_accuracy])

print('BEFORE TRAINING', model.get_weights())


# ======================================================================================================================
# Data Generators
# ======================================================================================================================

# ceil of the scalar x is THE SMALLER INTEGER i, such that i >= x
test_steps = np.ceil(test_data_size / batch_size)  # number of validation and testing batches (same size)
print('test_steps', test_steps)

# (number of batches) -1, because batches start from 0
# test_batch_generator = batch_generator(x_test_filename, '', batch_size, test_steps - 1)  # testing batch generator
test_generator = DataGenerator(x_test_filename, '', test_steps, batch_size=batch_size, shuffle=False)


# ======================================================================================================================
# Model Loading
# ======================================================================================================================

# save_all_weights | load_all_weights: saves model and optimizer weights (save_weights and save)
# load_status = model.load_weights("pretrained_models\\fulltext_model_weights.h5")  # sentences_model_weights.h5
load_status = model.load_weights(args.pretrained_model_path)  # sentences_model_weights.h5
#

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
# y_pred = model.predict(x=test_batch_generator, steps=test_steps)  # steps=validation_steps, because it does not read the last batch
y_pred = model.predict(x=test_generator)
print(y_pred)
print('\nY_PRED SHAPE', np.array(y_pred, dtype=object).shape)


# ======================================================================================================================
# Evaluation
# ======================================================================================================================

# traditional evaluation the model's performance
traditional_evaluation.evaluation(y_pred=y_pred, x_filename=x_filename, y_filename=y_filename)

# sequence evaluation of the model's performance
sequence_evaluation.evaluation(y_pred, MAX_LEN, y_test_filename)


# ======================================================================================================================
# Count the total running time
# ======================================================================================================================

total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- %s running time ---" % total_time)
