import sys
import tables
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)


# ======================================================================================================================
# Argument parsing
# ======================================================================================================================

parser = ArgumentParser()

parser.add_argument("-m", "--mode", type=str,
                    help="choose which type of data to create (options are: train, validation or test)")

parser.add_argument("-sm", "--sentence_model", type=bool, default=False,
                    help="choose which data to load (options are: True for data split in sentences or False for whole title and abstracts)")

args = parser.parse_args()


# ======================================================================================================================
# Set batch size and file names in which pre-processed data will be saved
# ======================================================================================================================

if args.sentence_model:
    # Set batch size
    batch_size = 352#352  #224  # 1024  # set during pre-processing (set in file preprocessing.py)
    # Set Abstract + Title max word size (text longer than the number will be trancated)
    max_len = 40  # 70  # Used to match the data dimensions for both TRAIN and TEST data (avg = 16)

    if args.mode == 'train':
        # Define the file paths and names to save TRAIN data
        x_filename = 'data\\preprocessed_data\\x_TRAIN_SENTENC_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TRAIN_SENTENC_data_preprocessed'
    elif args.mode == 'validation':
        # Define the file paths and names to save VALIDATION data to tune model parameters
        x_filename = 'data\\preprocessed_data\\x_VALIDATION_SENTENC_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_VALIDATION_SENTENC_data_preprocessed'
    elif args.mode == 'test':
        # Define the file paths and names to save TEST data to evaluate the final model
        x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_data_preprocessed'
    else:
        print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
        sys.exit()
else:
    # Set batch size
    batch_size = 64  # 1024  # set during pre-processing (set in file preprocessing.py)
    # Set Abstract + Title max word size (text longer than the number will be trancated)
    max_len = 400  # 500  # Used to match the data dimensions for both TRAIN and TEST data (max_len = 2763, avg = 147)

    if args.mode == 'train':
        # Define the file paths and names to save TRAIN data
        x_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed'
    elif args.mode == 'validation':
        # Define the file paths and names to save VALIDATION data to tune model parameters
        x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'
    elif args.mode == 'test':
        # Define the file paths and names to save TEST data to evaluate the final model
        x_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TEST_data_preprocessed'
    else:
        print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
        sys.exit()


print("Βatch size", batch_size)
print("Maximum length of title and abstract in the whole dataset", max_len)


# ======================================================================================================================
# Load pre-processed X and y values in file in order to change the batch size efficiently
# ======================================================================================================================
import json
# load data from files that contain only the whole pre-processed data sequences without any padding
with open(x_filename+".txt", "r") as fp:
    X = json.load(fp)
with open(y_filename+".txt", "r") as fp:
    y = json.load(fp)
#print(X)
#print(y)



# ======================================================================================================================
# Necessary pre-processing for Bi-LSTM (in general for neural networks)
# ======================================================================================================================

# Find the maximum length of abstract in the whole dataset
# Max length of abstract (2871)
# Max length of title and abstract (2763)
# max_len = max(data['abstract'].apply(len))  # Max length of abstract
print('X SHAPE', pd.DataFrame(X).shape)  # Max length of title and abstract

for i in tqdm(range(0, len(X), batch_size)):
    '''
    Split the non padded data in batches and iterate through each batch in order to pad_sequences. In each new
    iteration the data produced by pad_sequences are saved in file and the next batch is loaded in the place of the
    previous in order to release memory.
    '''

    # Padding each sentence to have the same length - padding values are padded to the end
    # sequences: List of sequences (each sequence is a list of integers)
    # maxlen: maximum length of all sequences. If not provided, sequences will be padded to the length of the longest individual sequence.
    # padding: 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence
    # value: Float or String, padding value (defaults to 0) - the value that will be added to pad the sequences/texts
    X_batch = pad_sequences(sequences=X[i:i + batch_size], padding="post", maxlen=max_len, value=0)


    # do NOT pad_sequences and to_categorical for TEST DATA
    if 'TEST' not in x_filename:

        # Padding each sentence to have the same length - padding values are padded to the end
        # value: padding value is set to 0, because the padding value CANNOT be a keyphrase
        y_batch = pad_sequences(sequences=y[i:i + batch_size], padding="post", maxlen=max_len, value=0)

        #print('X SHAPE AFTER', np.array(X_batch, dtype=object).shape)
        #print('y SHAPE AFTER', np.array(y_batch, dtype=object).shape)


# ======================================================================================================================
# Convert y values to CATEGORICAL
# ======================================================================================================================

        # REQUIRED - transform y to categorical (each column is a value of y - like in one-hot-encoding, columns are the vocabulary)
        y_batch = [to_categorical(i, num_classes=2, dtype='int8') for i in y_batch]

        #print(y)

        #print('After processing, sample:', X[0])
    #    print('After processing, labels:', y_batch[0])
        #print('y SHAPE AFTER', np.array(y_batch, dtype=object).shape)
        #print('X SHAPE AFTER', np.array(X_batch, dtype=object).shape)


# ======================================================================================================================
# Write pre-processed TRAIN data to csv file
# ======================================================================================================================

    # Set the compression level
    filters = tables.Filters(complib='blosc', complevel=5)

    # Save X batches into file
    f = tables.open_file(x_filename+'.hdf', 'a')
    ds = f.create_carray('/', 'x_data'+str(i), obj=X_batch, filters=filters)
    ds[:] = X_batch
    #print(ds)
    f.close()

    if 'TEST' not in x_filename:  # do NOT write for TEST DATA
        # Save y batches into file
        f = tables.open_file(y_filename + '.hdf', 'a')
        ds = f.create_carray('/', 'y_data' + str(i), obj=y_batch, filters=filters)
        ds[:] = y_batch
        f.close()


    # free memory here in order to allow for bigger batches (not needed, but allows for bigger sized batches)
    X_batch = None
    y_batch = None


# ======================================================================================================================
# Read data in chunks
# ======================================================================================================================

# Read X batches from file (pre-processed)
with tables.File(x_filename+'.hdf', 'r') as h5f:
    x = h5f.get_node('/x_data'+str(1024)).read()  # get a specific chunk of data
    print(x)
    print('X SHAPE AFTER', np.array(x, dtype=object).shape)


if 'TEST' not in x_filename:  # write ONLY for TEST DATA
    # Read y batches from file (pre-processed)
    with tables.File(y_filename+'.hdf', 'r') as h5f:
        y = h5f.get_node('/y_data'+str(1024)).read()  # get a specific chunk of data
        print(y)
        print('y SHAPE AFTER', np.array(y, dtype=object).shape)
