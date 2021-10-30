import time
import sys
import tables  # load compressed data files
import numpy as np
import pandas as pd
from tf2crf import CRF
from numpy import load
import tensorflow as tf
import tensorflow.keras.backend as K
import sequence_evaluation
import traditional_evaluation
from datetime import timedelta
import matplotlib.pyplot as plt
from tensorflow import constant  # used to convert array/list to a Keras Tensor
from argparse import ArgumentParser
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model, Input
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import TensorBoard
from data_generator import DataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


pd.set_option('display.max_columns', None)


# count the running time of the program
start_time = time.time()

import os
# disable tensorflow GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
# set memory growth for GPUs True
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
'''

'''
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


tf.config.gpu.set_per_process_memory_fraction(0.75)
tf.config.gpu.set_per_process_memory_growth(True)


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3,072)])


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''


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

parser.add_argument("-sm", "--sentence_model", type=int, default=0,
                    help="choose which data to load (options are: True for sentence model or False for whole title and abstracts model)")

args = parser.parse_args()


# ======================================================================================================================
# Set data generators for batch training
# ======================================================================================================================

if args.sentence_model:
    # Set batch size, train and test data size
    batch_size = 256#352#224  # 1024  # set during pre-processing (set in file preprocessing.py)
    train_data_size = 4136306#4139868  # 4147964  [ THE NUMBER OF TRAIN SENTENCES\DOCS ]  # the total size of train data
    validation_data_size = 156519#156836  # 156836  [ THE NUMBER OF VALIDATION SENTENCES\DOCS ]  # the total size of test data
    #test_data_size = 155801#156085  # 156085  SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data

    # Set INPUT layer size
    MAX_LEN = 40#50  # 70  # max length of abstract and title (together) (full text train set: 2763)
else:
    # Set batch size,train and test data size
    batch_size = 64 #32  # 1024  # set during pre-processing (set in file preprocessing.py)
    train_data_size = 530390 #530809  # 530880  [ THE NUMBER OF TRAIN SENTENCES\DOCS ]  # the total size of train data
    validation_data_size = 20000  # 20064  # 20000  [ THE NUMBER OF VALIDATION SENTENCES\DOCS ]  # the total size of test data
    #test_data_size = 20000  # 20064  # 20000  SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data

    # Set INPUT layer size
    MAX_LEN = 400  # 500  # 70  # max length of abstract and title (together) (full text train set: 2763)


# Set embedding size, OUTPUT layer size
VECT_SIZE = 100  # the GloVe vector size
number_labels = 2  # 2 labels, keyword (KP) and Non-keyword (Non-KP)

doc_vocab = 321352#321490    #298383  # 344858    # 323895    # 100003  # the vocabulary of the train dataset

print('MAX_LEN of text', MAX_LEN)
print('VECT_SIZE', VECT_SIZE)
print('VOCABULARY', doc_vocab)


# ======================================================================================================================
# Define train/test data file names
# ======================================================================================================================

if args.sentence_model:
    # [SENTENCES ABSTRACT TEXT - train_data_size = 4136306] Define the file paths and names for TRAINING data
    x_train_filename = 'data\\preprocessed_data\\x_TRAIN_SENTENC_data_preprocessed.hdf'
    y_train_filename = 'data\\preprocessed_data\\y_TRAIN_SENTENC_data_preprocessed.hdf'

    # [SENTENCES ABSTRACT TEXT - validation_data_size = 156519] Define the file paths and names for VALIDATION data to tune model parameters
    x_validate_filename = 'data\\preprocessed_data\\x_VALIDATION_SENTENC_data_preprocessed.hdf'
    y_validate_filename = 'data\\preprocessed_data\\y_VALIDATION_SENTENC_data_preprocessed.hdf'
else:
    # [FULL ABSTRACT TEXT - train_data_size = 530390] Define the file paths and names for TRAINING data
    x_train_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed.hdf'
    y_train_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed.hdf'

    # [FULL ABSTRACT TEXT - validation_data_size = 20000] Define the file paths and names for VALIDATION data to tune model parameters
    x_validate_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed.hdf'
    y_validate_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed.hdf'


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

    if not y_filename == '':    # for TEST data read only the x values
        # Read y batches for testing from file (pre-processed)
        with tables.File(y_filename, 'r') as h5f:
            y = h5f.get_node('/y_data' + str(batch_number)).read()  # get a specific chunk of data

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

# Generators
training_generator = DataGenerator(x_train_filename, y_train_filename, steps_per_epoch, batch_size=batch_size, shuffle=False)
validation_generator = DataGenerator(x_validate_filename, y_validate_filename, validation_steps, batch_size=batch_size, shuffle=False)
test_generator = DataGenerator(x_test_filename, '', test_steps, batch_size=batch_size, shuffle=False)

# ======================================================================================================================
# Define f1-score to monitor during training and save the best model with Checkpoint
# ======================================================================================================================

def load_y_val(y_file_name, batch_size, number_of_batches):
    """
    Load y_test for validation
    :param y_file_name: the file name that contains pre-processed data of y_test
    :param batch_size: the size of each batch
    :param number_of_batches: the total number of batches
    :return: return y_test (y_test_flat) for validation
    """
    y_val_batches = []  # save y_test for validation
    current_batch_number = 0  # the identifier used for each batch of data (ex. 0, 10000, 20000, 30000, etc.)
    while True:
        # Read X batches for testing from file (pre-processed)
        with tables.File(y_file_name, 'r') as h5f:
            y_val_batches.append(h5f.get_node('/y_data' + str(current_batch_number)).read())  # get a specific chunk of data

        # calculate the identifier of each batch of data
        if current_batch_number < batch_size * number_of_batches:
            current_batch_number += batch_size
        else:
            y_val_flat = [y_label for y_batch in y_val_batches for y_label in y_batch]  # flatten the y_test (20000, 2881, 2)
            print('y_test SHAPE AFTER', np.array(y_val_flat, dtype=object).shape)
            return y_val_flat


def pred2label(all_abstract_preds):
    '''
    Converts prediction set and test/validation set from two columns (one for each label value)
    to just one column with the number of the corresponding label
    [ initial array: [1, 0] => final array: [0] ]   -   [ initial array: [0, 1] => final array: [1] ]
    :param all_abstract_preds: array with predictions or test/validation set [documents/abstracts, number of words]
    :return: flattened array that contains the prediction for each word [number of total words of all abstracts]
    '''
    preds = []
    for abstract_preds in all_abstract_preds:
        # the position of the max value is corresponding to the actual label value (0: Non-KP, 1: KP)
        preds.extend([np.argmax(word_pred) for word_pred in abstract_preds])
    return preds


y_val = load_y_val(y_validate_filename, batch_size, validation_steps - 1)  # load y_test in memory
y_val = pred2label(y_val)  # convert y_test from categorical (two columns, 1 for each label) to a single value label


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # y_pred = self.model.predict(x=validation_batch_generator, steps=validation_steps)  # steps=validation_steps, because it does not read the last batch
        y_pred = self.model.predict(x=validation_generator)
        y_pred = pred2label(y_pred)  # convert y_test from categorical (two columns, 1 for each label) to a single value label
        # print("Epoch: {} F1-score: {:.2%}".format(epoch, f1_score(y_test, y_pred, pos_label=1)))
        logs.update({'val_f1score': f1_score(y_val, y_pred, pos_label=1)})
   #     logs['val_f1score'] = f1_score(y_test, y_pred, pos_label=1)
        super().on_epoch_end(epoch, logs)


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

# Model definition
inpt = Input(shape=(MAX_LEN,))

# doc_vocab: vocabulary - number of words - of the train dataset
model = Embedding(doc_vocab, output_dim=100, input_length=MAX_LEN,  # n_words + 2 (PAD & UNK)
                  weights=[embedding_matrix],  # use GloVe vectors as initial weights
                  mask_zero=True, trainable=True, activity_regularizer=l1(0.00000001))(inpt)

model = Bidirectional(LSTM(units=100, return_sequences=True, activity_regularizer=l1(0.0000000001), recurrent_constraint=max_norm(2)))(model)  # input_shape=(1, MAX_LEN, VECT_SIZE)
model = Dense(number_labels, activation=None)(model)
crf = CRF()  # CRF layer
out = crf(model)  # output
model = Model(inputs=inpt, outputs=out)


# set learning rate
#lr_rate = InverseTimeDecay(initial_learning_rate=0.05, decay_rate=4, decay_steps=steps_per_epoch)
# lr_rate = ExponentialDecay(initial_learning_rate=0.01, decay_rate=0.5, decay_steps=10000)


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    # epochs_drop = 1.0  # how often to change the learning rate
    lrate = initial_lrate / (1 + drop * epoch)
    '''
    lrate=[0.01, 0.0075, 0.005, 0.0025, 0.001]
    lrate = lrate[epoch]
    '''
    return lrate


lrate = LearningRateScheduler(step_decay)

# set optimizer
# decay=learning_rate / epochs
opt = SGD(learning_rate=0.0, momentum=0.9, clipvalue=5.0)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]

# compile Bi-LSTM-CRF
model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])

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


my_callbacks = [
    lrate,  # learning rate scheduler
    # save the weights of the model with the best f1-score after each epoch
    tf.keras.callbacks.ModelCheckpoint(filepath='pretrained_models\\checkpoint\\model.{epoch:02d}.h5',  # -{val_f1score:.2f}   val_loss
                                       save_weights_only=True,
                                #       monitor='val_f1score',  # val_loss
                                #       mode='max',  # min
                                       save_best_only=False),
    LRTensorBoard(log_dir="/tmp/tb_log"),  # report learning rate after each epoch
    PredictionCallback()
]


# Train model
history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=5, callbacks=my_callbacks, verbose=2)

model.summary()

# print('AFTER TRAINING', model.get_weights())


# ======================================================================================================================
# Predict on validation data
# ======================================================================================================================

print('\nPredicting...')
# y_pred = model.predict(x=test_batch_generator, steps=test_steps)  # steps=validation_steps, because it does not read the last batch
y_pred = model.predict(x=test_generator)


# ======================================================================================================================
# Evaluation
# ======================================================================================================================

# traditional evaluation the model's performance
traditional_evaluation.evaluation(y_pred=y_pred, x_filename=x_filename, y_filename=y_filename)

# sequence evaluation of the model's performance
sequence_evaluation.evaluation(y_pred, MAX_LEN, y_test_filename)


# ======================================================================================================================
# Model Saving
# ======================================================================================================================

# Save the model (weights)
# save_all_weights | load_all_weights: saves model and optimizer weights (save_weights and save)
model.save_weights("pretrained_models\\fulltext_model_weights.h5")  # sentences_model_weights.h5

# Save the model (architecture, loss, metrics, optimizer state, weights)
model.save('pretrained_models\\fulltext_bi_lstm_crf_dense_linear.h5')  # sentences_bi_lstm_crf_dense_linear.h5
'''
# Load the model
from keras.models import load_model
model = load_model('pretrained_models\\fulltext_bi_lstm_crf_dense_linear.h5', custom_objects={'CRF': CRF(number_labels), 'num_classes': number_labels})  #  , 'loss': crf.loss, 'metrics': [crf.accuracy]
'''


# ======================================================================================================================
# Count the total running time
# ======================================================================================================================

total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- %s running time ---" % total_time)


# ======================================================================================================================
# Track model loss per epoch
# ======================================================================================================================

with open("pretrained_models\\Results.txt", "a") as myfile:  # Write above print into output file
    myfile.write("f1-score after each epoch: " + str(history.history['val_f1score']) + '\n')
    myfile.write("learning rate after each epoch: " + str(history.history['lr']))

print('\nf1-score after each epoch: ', history.history['val_f1score'])
print('learning rate after each epoch: ', history.history['lr'])
print('loss: ', history.history['loss'])
print('accuracy: ', history.history['accuracy'])
print('val_loss: ', history.history['val_loss'])
print('val_accuracy: ', history.history['val_accuracy'])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('pretrained_models\\model_loss_per_epoch.png')  # save the plot of model's loss per epoch to file


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('pretrained_models\\model_accuracy_per_epoch.png')  # save the plot of model's loss per epoch to file


# ======================================================================================================================
# Plot the layer architecture of LSTM
# ======================================================================================================================

plot_model(model, "schemas\\bi-lstm-crf_architecture.png", show_shapes=True)
