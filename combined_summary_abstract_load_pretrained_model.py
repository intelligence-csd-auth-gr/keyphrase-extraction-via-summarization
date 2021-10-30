import ast  # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
import sys
import time
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
                    help="select the test set to evaluate the model (options are: nus, acm or semeval)")

parser.add_argument("-pmp", "--pretrained_model_path", type=str,
                    help="the path and the name of the pretrained model")

parser.add_argument("-sm", "--sentence_model", type=int, default=0,
                    help="choose which data to load (options are: '1' for sentence model or '0' for whole title and abstracts model)")

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
    #test_data_size = 244  # 20000  SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data

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

# Summarization of abstract and fulltext
if args.select_test_set=="nus":
    # [ test_data_size = 211 ]
    test_data_size = 211
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_NUS_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_NUS_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename_summarization = 'data\\preprocessed_data\\summarization_experiment\\x_NUS_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_NUS_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
elif args.select_test_set=="acm":
    # [ test_data_size = 2304 ]
    test_data_size = 2304
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_ACM_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_ACM_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename_summarization = 'data\\preprocessed_data\\summarization_experiment\\x_ACM_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_ACM_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
elif args.select_test_set=="semeval":
    # [ test_data_size = 244 ]
    test_data_size = 244
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_SEMEVAL_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_SEMEVAL_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename_summarization = 'data\\preprocessed_data\\summarization_experiment\\x_SEMEVAL_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_SEMEVAL_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
else:
    print('WRONG ARGUMENTS! - please fill the argument "-sts" or "--select_test_set" with one of the proper values (acm, nus or semeval)')
    sys.exit()


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
y_pred_summaries = model.predict(x=test_generator)
print(y_pred_summaries)
print('\nY_PRED SHAPE', np.array(y_pred_summaries, dtype=object).shape)
















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
    #test_data_size = 244  # 20000  SEE BELOW [ THE NUMBER OF TEST SENTENCES\DOCS ]  # the total size of test data

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
if args.select_test_set=="nus":
    # [ test_data_size = 211 ]
    test_data_size = 211
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_preprocessed_TEXT'
elif args.select_test_set=="acm":
    # [ test_data_size = 2304 ]
    test_data_size = 2304
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_TEST_vectors.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_TEST_vectors'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_preprocessed_TEXT'
elif args.select_test_set=="semeval":
    # [ test_data_size = 244 ]
    test_data_size = 244
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_SEMEVAL_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_SEMEVAL_FULL_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_SEMEVAL_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_SEMEVAL_FULL_ABSTRACT_preprocessed_TEXT'
else:
    print('WRONG ARGUMENTS! - please fill the argument "-sts" or "--select_test_set" with one of the proper values')
    sys.exit()


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
inpt = Input(shape=(None,))

# doc_vocab: vocabulary - number of words - of the train dataset
model = Embedding(doc_vocab, output_dim=100, input_length=None,  # n_words + 2 (PAD & UNK)
                  weights=[embedding_matrix],  # use GloVe vectors as initial weights
                  mask_zero=True, trainable=True, activity_regularizer=l1(0.0000001))(inpt)  # name='word_embedding'

# recurrent_dropout=0.1 (recurrent_dropout: 10% possibility to drop of the connections that simulate LSTM memory cells)
model = Bidirectional(LSTM(units=100, return_sequences=True, activity_regularizer=l1(0.0000000001), recurrent_constraint=max_norm(2)))(model)  # input_shape=(1, MAX_LEN, VECT_SIZE)
model = Dense(number_labels, activation=None)(model)  # activation='linear' (they are the same)
crf = CRF()  # CRF layer { SHOULD I SET -> number_labels+1 (+1 -> PAD) }
out = crf(model)  # output
model = Model(inputs=inpt, outputs=out)

# set optimizer
opt = SGD(learning_rate=0.0, momentum=0.9, clipvalue=5.0)  # clipvalue (Gradient Clipping): clip the gradient to [-5 to 5]

# compile Bi-LSTM-CRF
model.compile(optimizer=opt, loss=crf.loss, metrics=[crf.accuracy])

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
y_pred_abstract = model.predict(x=test_generator)
print(y_pred_abstract)
print('\nY_PRED SHAPE', np.array(y_pred_abstract, dtype=object).shape)


# ======================================================================================================================
# Create combined y_pred from abstract and summary predicted keyphrases
# ======================================================================================================================
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
        doc_preds = [np.argmax(word_pred) for word_pred in abstract_preds]
        preds.append(doc_preds)
    return preds


# print('BEFORE y_pred', y_pred)
y_pred_abstract = pred2label(y_pred_abstract)
y_pred_summaries = pred2label(y_pred_summaries)
# print('AFTER y_pred', y_pred)


# ======================================================================================================================
# Extract keyphrases from the predicted set
# ======================================================================================================================
x_test = pd.read_csv(x_filename, encoding="utf8")
x_test['abstract'] = x_test['abstract'].map(ast.literal_eval)

pred_keyphrase_list_abstract = []  # save all predicted keyphrases
for doc_index, doc_prediction in enumerate(y_pred_abstract):  # iterate through predictions for documents
    document_keyphrases = []  # save the keyphrases of a document
    consecutive_keywords = []  # save consecutive keywords that form a keyphrase
    for word_index, word_prediction in enumerate(doc_prediction):  # iterate through predictions for WORDS of documents
        if word_index >= len(x_test['abstract'][doc_index]):
            break  # check if the abstract reached to an end (padding adds more dummy words non existing in real abstract)
        if word_index:  # check if this is the FIRST WORD in the abstract [to avoid negative index value]
            if doc_prediction[word_index - 1]:  # check if the previous word is a keyword
                if word_prediction:  # check if the current word is a keyword
                    #                        print(x_test['abstract'][doc_index])
                    #                        print(x_test['abstract'][doc_index][word_index])
                    consecutive_keywords.append(x_test['abstract'][doc_index][word_index])
            else:
                if len(consecutive_keywords):  # save keyword list if exists (not empty list)
                    document_keyphrases.append(consecutive_keywords)
                consecutive_keywords = []  # re-initialize (empty) list
                if word_prediction:  # check if the current word is a keyword
                    consecutive_keywords.append(x_test['abstract'][doc_index][word_index])
        else:  # save the FIRST WORD of the abstract if it is a keyword
            if word_prediction:  # check if the current word is a keyword
                #               print('HEREEEE', doc_index, word_index)
                #               print(x_test['abstract'][doc_index])
                consecutive_keywords.append(x_test['abstract'][doc_index][word_index])

    if len(consecutive_keywords):  # save the keywords that occur in the END of the abstract, if they exist (not empty list)
        document_keyphrases.append(consecutive_keywords)

    pred_keyphrase_list_abstract.append(document_keyphrases)



x_test_summar = pd.read_csv(x_filename_summarization, encoding="utf8")
x_test_summar['abstract'] = x_test_summar['abstract'].map(ast.literal_eval)

pred_keyphrase_list_summaries = []  # save all predicted keyphrases
for doc_index, doc_prediction in enumerate(y_pred_summaries):  # iterate through predictions for documents
    document_keyphrases = []  # save the keyphrases of a document
    consecutive_keywords = []  # save consecutive keywords that form a keyphrase
    for word_index, word_prediction in enumerate(doc_prediction):  # iterate through predictions for WORDS of documents
        if word_index >= len(x_test_summar['abstract'][doc_index]):
            break  # check if the abstract reached to an end (padding adds more dummy words non existing in real abstract)
        if word_index:  # check if this is the FIRST WORD in the abstract [to avoid negative index value]
            if doc_prediction[word_index - 1]:  # check if the previous word is a keyword
                if word_prediction:  # check if the current word is a keyword
                    #                        print(x_test['abstract'][doc_index])
                    #                        print(x_test['abstract'][doc_index][word_index])
                    consecutive_keywords.append(x_test_summar['abstract'][doc_index][word_index])
            else:
                if len(consecutive_keywords):  # save keyword list if exists (not empty list)
                    document_keyphrases.append(consecutive_keywords)
                consecutive_keywords = []  # re-initialize (empty) list
                if word_prediction:  # check if the current word is a keyword
                    consecutive_keywords.append(x_test_summar['abstract'][doc_index][word_index])
        else:  # save the FIRST WORD of the abstract if it is a keyword
            if word_prediction:  # check if the current word is a keyword
                #               print('HEREEEE', doc_index, word_index)
                #               print(x_test['abstract'][doc_index])
                consecutive_keywords.append(x_test_summar['abstract'][doc_index][word_index])

    if len(consecutive_keywords):  # save the keywords that occur in the END of the abstract, if they exist (not empty list)
        document_keyphrases.append(consecutive_keywords)

    pred_keyphrase_list_summaries.append(document_keyphrases)


y_pred = []
for indx, y_abstract in enumerate(pred_keyphrase_list_abstract):
    print('y_abstract: ', y_abstract)
    print('y_pred_summaries BEFORE: ', pred_keyphrase_list_summaries[indx])
    y_abstract.extend(pred_keyphrase_list_summaries[indx])
    print('y_pred_summaries AFTER: ', y_abstract)
    y_pred.append(y_abstract)
print(y_pred)


# ======================================================================================================================
# Combine ABSTRACT + SUMMARIES
# ======================================================================================================================

# needed for extraction f1-score - gold keyphrases should be checked if they exist with both abstract and summary
for doc_index, test_summar in enumerate(x_test_summar['abstract']):
    x_test_summar['abstract'].iat[doc_index] = ' '.join(x_test['abstract'][doc_index]) + ' ' + ' '.join(test_summar)


# ======================================================================================================================
# Evaluation
# ======================================================================================================================
gold_keyphrases = pd.read_csv(y_filename, encoding="utf8")
gold_keyphrases = gold_keyphrases['keyword'].map(ast.literal_eval)

# traditional evaluation the model's performance
traditional_evaluation.evaluation(y_pred=y_pred, y_test=gold_keyphrases, x_test=x_test_summar, x_filename=x_filename)

# sequence evaluation of the model's performance
#sequence_evaluation.evaluation(y_pred, MAX_LEN, y_test_filename)


# ======================================================================================================================
# Count the total running time
# ======================================================================================================================

total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- %s running time ---" % total_time)
