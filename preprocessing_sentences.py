import json
import string
import tables
import numpy as np
import pandas as pd
from string import digits
from pandas import json_normalize
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer


pd.set_option('display.max_columns', None)

# ======================================================================================================================
# Set batch size and file names in which pre-processed data will be saved
# ======================================================================================================================

# reading the initial JSON data using json.load()
file = 'data\\kp20k_training.json'  # TRAIN data
# file = 'data\\kp20k_validation.json'  # VALIDATION data to tune model parameters
# file = 'data\\kp20k_testing.json'  # TEST data to evaluate the final model


# Define the file paths and names to save TRAIN data
x_filename = 'data\\preprocessed_data\\x_TRAIN_SENTENC_data_preprocessed'
y_filename = 'data\\preprocessed_data\\y_TRAIN_SENTENC_data_preprocessed'


'''
# Define the file paths and names to save VALIDATION data to tune model parameters
x_filename = 'data\\preprocessed_data\\x_VALIDATION_SENTENC_data_preprocessed'
y_filename = 'data\\preprocessed_data\\y_VALIDATION_SENTENC_data_preprocessed'
'''


'''
# Define the file paths and names to save TEST data to evaluate the final model
x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed'
'''

x_text_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_preprocessed_TEXT'  # save preprosessed text for TEST data
y_text_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_preprocessed_TEXT'  # save preprosessed keyphrases for TEST data


# Define the number of lines to read
batch_size = 1024   # 10000


# ======================================================================================================================
# Read data
# ======================================================================================================================

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)

'''
DATA IS ALREADY CLEAN

# Clean missing data
data = data.dropna()
data.reset_index(drop=True, inplace=True)  # needed when we sample the data in order to join dataframes
print(data)
'''

'''
# get a sample of the whole dataset (for development ONLY)
data = data.sample(n=1024, random_state=42)
data.reset_index(inplace=True)  # NEEDED if sample is enabled in order to use enumerate in the for loop below
'''


# ======================================================================================================================
# Split keyphrases list of keyphrases from string that contains all the keyphrases
# ======================================================================================================================

for index, keywords in enumerate(data['keyword']):
    data['keyword'].iat[index] = keywords.split(';')  # split keywords to separate them from one another


# ======================================================================================================================
# Combine the title and abstract (+ remove '\n')
# ======================================================================================================================

# tokenize key-phrases and keep them categorized by document
for index, abstract in enumerate(data['abstract']):
    title_abstract = data['title'][index] + '. ' + abstract  # combine title + abstract
    # remove '\n'
    title_abstract = title_abstract.replace('\n', ' ')
#    print('title_abstract_mainBody', title_abstract)

    data['abstract'].iat[index] = title_abstract


# ======================================================================================================================
# Split text to sentences so that each sentence corresponds to one document
# ======================================================================================================================

# Split text to sentences
data['abstract'] = data['abstract'].apply(sent_tokenize)
print(data['abstract'][0])

# Split each row containing list of sentences to a sentence per row (one sentence is considered as one document)
data = data.explode('abstract')
print(data)

# reset index as explode results in multiple rows having the same index
data.reset_index(inplace=True)

'''
print(data)
import sys
sys.exit()  # terminate program to save the number of total documents-rows-sentences
'''


# ======================================================================================================================
# Remove punctuation
# ======================================================================================================================

# keep punctuation that might be useful                     -  !"#$%&'()*+,./:;<=>?@[\]^_`{|}~
#keep_punctuation = ['<', '=', '>', '^', '{', '|', '}', '/', '%', '$', '*', '&']
#punctuation = [punct for punct in string.punctuation if punct not in keep_punctuation]
punctuation = [punct for punct in string.punctuation]
print(punctuation)


# remove special characters and punctuations
def remove_punct(text):
    table = str.maketrans(dict.fromkeys(punctuation))  # OR {key: None for key in string.punctuation}
    clean_text = text.translate(table)
    return clean_text


def keyword_remove_punct(text):
    table = str.maketrans(dict.fromkeys(punctuation))  # OR {key: None for key in string.punctuation}
    clean_text = []
    for keyw in text:
        clean_text.append(keyw.translate(table))
    return clean_text


# REMOVE
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# <=>\^{|}          /
# % $ -             & *

# remove punctuation
data['abstract'] = data['abstract'].apply(remove_punct)
data['keyword'] = data['keyword'].apply(keyword_remove_punct)
# print(data)


# ======================================================================================================================
# Tokenize each sentence
# ======================================================================================================================

remove_digits = str.maketrans('', '', digits)  # remove digits


def tokenize_stem_lowercase(text):
    '''
    Toekenize, stem and convert to lower case the text of documents
    :param text: text of a specific document
    :return: formatted text
    '''
    formatted_text = []
    words = word_tokenize(text)  # tokenize document text
    for word_token in words:  # get words of all keyphrases in a single list
        word_token = word_token.translate(remove_digits)  # remove digits
        formatted_text.append(word_token.lower())  # DO NOT STEM TEXT WORDS TO TRAIN THE CLASSIFIER
    return formatted_text


# tokenize text
data['abstract'] = data['abstract'].apply(tokenize_stem_lowercase)
print(data['abstract'])
print('abstract finish')


# stem, tokenize and lower case keyphrases and keep them categorized by document
for index, list_of_keyphrases in enumerate(data['keyword']):
    keyphrases_list = []
    for keyphrase in list_of_keyphrases:  # get words of all keyphrases in a single list
        if len(keyphrase.strip()):  # check if the keyphrase is empty
            tokens = word_tokenize(keyphrase)  # tokenize
            keyphrases_list.append([Stemmer('porter').stem(keyword.translate(remove_digits).lower()) for keyword in tokens])  # stem + lower case
    data['keyword'].iat[index] = keyphrases_list
#    print('THESE ARE THE KEYPHRASE LIST', len(keyphrases_list), keyphrases_list)


# ======================================================================================================================
# Write pre-processed keyphrases to csv file (if the data are the TEST DATA)
# ======================================================================================================================

if x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed':  # if the data are the TEST DATA
    data['abstract'].to_csv(x_text_filename, index=False)  # save the preprocessed document text
    data['keyword'].to_csv(y_text_filename, index=False)  # save the preprocessed keyphrases


# ======================================================================================================================
# GloVe vectors
# ======================================================================================================================

# gloveFile = 'GloVe\\glove.twitter.27B\\glove.twitter.27B.100d.txt'
gloveFile = 'GloVe\\glove.6B\\glove.6B.100d.txt'

print("Loading Glove Model")
glove_model = {}
with open(gloveFile, 'r', encoding="utf8") as f:
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array(splitLine[1:], dtype='float32')
        glove_model[word] = embedding


# get the vector of a word
def get_glove_vec(word_vec):
    '''
    Get the vector of a word
    :param word_vec: a given word
    :return: the GloVe vector of given word
    '''
    if word_vec == 'PADDING_VALUE':  # return the padding vector (size=100)
        padding_vector = np.zeros(100, dtype='float64')  # create a vector of zeros' to use as padding value
        return padding_vector  # np.array(padding_vector, dtype='float64')

    embedding_vector = glove_model.get(word_vec)  # get the vector of a word, if exists

    if embedding_vector is not None:  # we found the word - add that words vector to the matrix
        return embedding_vector
    else:  # given word doesn't exist, assign the average vector of all word vectors of GloVe (size=100)
        avg_vector = [-1.55611530e-01, -3.93543998e-03, -3.25425752e-02, -6.28335699e-02,
                      -4.78157075e-03, -1.84617653e-01, -4.94439378e-02, -1.80521324e-01,
                      -3.35793793e-02, -1.94202706e-01, -6.56424314e-02, 3.70132737e-02,
                      6.60830796e-01, -7.80616794e-03, -1.95153028e-04, 9.07416344e-02,
                      8.08839127e-02, 7.30239078e-02, 2.30256692e-01, 9.59603861e-02,
                      1.10939644e-01, 4.32463065e-02, 6.52063936e-02, -6.03170432e-02,
                      -2.05838501e-01, 7.50285745e-01, 1.29861072e-01, 1.11144960e-01,
                      -3.51636916e-01, 2.48395074e-02, 2.55199522e-01, 1.77181944e-01,
                      4.26653862e-01, -2.19325781e-01, -4.53459173e-01, -1.41409159e-01,
                      -8.41409061e-03, 1.01715224e-02, 6.76619932e-02, 1.28284395e-01,
                      6.85148776e-01, -1.77478697e-02, 1.28944024e-01, -7.42785260e-02,
                      -3.58294964e-01, 9.97241363e-02, -5.09560928e-02, 5.47902798e-03,
                      6.24788366e-02, 2.89107800e-01, 3.06909740e-01, 9.53846350e-02,
                      -8.97585154e-02, -3.03416885e-02, -1.80602595e-01, -1.63290858e-01,
                      1.23387389e-01, -1.73964277e-02, -6.13645613e-02, -1.17096789e-01,
                      1.49090782e-01, 1.17921308e-01, 1.05730975e-02, 1.33317500e-01,
                      -1.94899425e-01, 2.25606456e-01, 2.08363295e-01, 1.73583731e-01,
                      -4.40407135e-02, -6.87221363e-02, -1.83684096e-01, 7.04482123e-02,
                      -6.98078275e-02, 2.02260930e-02, 3.70468129e-03, 1.96141958e-01,
                      1.96837828e-01, 1.27971312e-02, 4.36565094e-02, 1.42354667e-01,
                      -3.62371027e-01, -1.10718250e-01, -4.84273471e-02, 4.64920104e-02,
                      -1.09924808e-01, -1.34851769e-01, 1.89310268e-01, -3.97192866e-01,
                      5.38146198e-02, -1.40333608e-01, 5.22745401e-02, 1.40163332e-01,
                      1.00092500e-01, 6.39176890e-02, 5.10458164e-02, 8.40307549e-02,
                      1.05783986e-02, 2.15598941e-01, -1.54302031e-01, 1.49716333e-01]
        return np.array(avg_vector, dtype='float64')  # return the average vector of all word vectors of GloVe


'''   
# Get the average vector of all GloVe vectors

# Get number of vectors and hidden dim
with open(gloveFile, 'r', encoding="utf8") as f:
    for i, line in enumerate(f):
        pass
n_vec = i + 1
hidden_dim = len(line.split(' ')) - 1

vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

with open(gloveFile, 'r', encoding="utf8") as f:
    for i, line in enumerate(f):
        vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

average_vec = np.mean(vecs, axis=0)
print('average_vec: ', average_vec)
'''


# ======================================================================================================================
# Get the GloVe representation of documents
# ======================================================================================================================

# Convert each title and abstract to list of GloVe vectors
X = []
for index, abstract in enumerate(data['abstract']):
    abstract_vectors = []  # saves GloVe vectors for one title + abstract

    # convert words in abstract to GloVe vectors
    for word in abstract:
        abstract_vectors.append(get_glove_vec(word))

    # save GloVe vectors for each title + abstract
    X.append(abstract_vectors)


# ======================================================================================================================
# Give labels to each word of Title and Abstract - keyword (KP) or Non-keyword (Non-KP)
# ======================================================================================================================

# Convert Tag/Label to tag_index
count_KP = 0  # count the number of keyphrases (used for cost-sensitive learning in training of Bi-LSTM-CRF)
count_NON_KP = 0  # count the number of NON keyphrases (used for cost-sensitive learning in training of Bi-LSTM-CRF)
y = []  # list of lists that contain the labels (keyword (KP), Non-keyword (Non-KP)) for each word of the abstract
for index, abstract in enumerate(data['abstract']):
    abstract_word_labels = [0] * len(abstract)  # keep the labels of the all the words contained in a single title and abstract text (initialize with 0s - Non-KP value)
#    print('abstract_word_labels', abstract_word_labels, 'len(abstract)', len(abstract))

    # add labels for words in abstract
    for i, word in enumerate(abstract):  # check the next word of keyphrase with the next word of abstract until the full keyphrase if found
        for keyphrase in data['keyword'][index]:
#            print('word == keyphrase[0]', Stemmer('porter').stem(word), '==', keyphrase[0])
#            print(keyphrase, '\n', len(keyphrase))
            if Stemmer('porter').stem(word) == keyphrase[0]:  # locate the beginning of a keyphrase in the abstract
                match_count = 1  # how many keywords of a keyphrase match to sequential words in abstract (start with 1 as the first word already matched)
                for j in range(1, len(keyphrase)):  # check the words after the first matching keyword
#                    print('HALF SECOND\ni + j < len(abstract)', i + j, '<=', len(abstract))
                    if i + j < len(abstract):  # check if the index get ouf of abstract bounds
#                        print('SECOND\nabstract[i + j] == keyphrase[j]', Stemmer('porter').stem(abstract[i + j]), '==', keyphrase[j])
                        if Stemmer('porter').stem(abstract[i + j]) == keyphrase[j]:  # the word is part of a keyphrase
                            match_count += 1
                        else:
                            break  # a word in the abstract does not match with the words in keyphrase
                    else:
                        break  # reached the end of the abstract (prevent out of bounds)
#                print('THIRD\nmatch_count == len(keyphrase)', match_count, '==', len(keyphrase))
                if match_count == len(keyphrase):
                    for x in range(len(keyphrase)):
                        abstract_word_labels[i + x] = 1
                    count_KP += 1
                    break  # end iteration when a keyphrase is found to avoid checking the rest keyphrases (it will convert the labels of the keyphrase in abstract to Non-KP)
                else:  # the word/phrase is not a keyphrase
                    # abstract_word_labels[i] = 0  # not needed as abstract_word_labels is initialized with 0s
                    count_NON_KP += 1
            else:  # the word is not a keyphrase
                if not abstract_word_labels[i]:  # check if the current word has been annotated as keyword to avoid changing its value
                    # abstract_word_labels[i] = 0  # not needed as abstract_word_labels is initialized with 0s
                    count_NON_KP += 1
#    print('AFTER abstract_word_labels', abstract_word_labels)
    # save labels for each title + abstract
    y.append(abstract_word_labels)

print('KP count: ', count_KP, '\nNON-KP count: ', count_NON_KP)
'''
KP count:  7862742 
NON-KP count:  75909213
'''


# ======================================================================================================================
# Necessary pre-processing for Bi-LSTM (in general for neural networks)
# ======================================================================================================================

# Find the maximum length of abstract in the whole dataset
# Max length of sentence in title and abstract (348)
# max_len = max(data['abstract'].apply(len))  # Max length of abstract
print('X SHAPE', pd.DataFrame(X).shape)  # Max length of title and abstract

# Set Abstract + Title max word size (text longer than the number will be trancated)
# max_len = 348  # Maximum length of abstract in the TRAIN data
max_len = 70  # Used to match the data dimensions for both TRAIN and TEST data
print("Maximum length of title and abstract in the whole dataset", max_len)

for i in range(0, len(X), batch_size):
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
    X_batch = pad_sequences(sequences=X[i:i + batch_size], padding="post", maxlen=max_len, value=get_glove_vec("PADDING_VALUE"))


    # Padding each sentence to have the same length - padding values are padded to the end
    # value: padding value is set to 0, because the padding value CANNOT be a keyphrase
    y_batch = pad_sequences(sequences=y[i:i + batch_size], padding="post", maxlen=max_len, value=0)

    print('X SHAPE AFTER', np.array(X_batch, dtype=object).shape)
    print('y SHAPE AFTER', np.array(y_batch, dtype=object).shape)


# ======================================================================================================================
# Convert y values to CATEGORICAL
# ======================================================================================================================

    # REQUIRED - transform y to categorical (each column is a value of y - like in one-hot-encoding, columns are the vocabulary)
    y_batch = [to_categorical(i, num_classes=2, dtype='int8') for i in y_batch]

    #print(y)

    #print('After processing, sample:', X[0])
#    print('After processing, labels:', y_batch[0])
    print('y SHAPE AFTER', np.array(y_batch, dtype=object).shape)
    #print('X SHAPE AFTER', np.array(X_batch, dtype=object).shape)


# ======================================================================================================================
# Write pre-processed TRAIN data to csv file
# ======================================================================================================================

    # Set the compression level
    filters = tables.Filters(complib='blosc', complevel=5)

    # Save X batches into file
    f = tables.open_file(x_filename+'.hdf', 'a')
    ds = f.create_carray('/', 'x_data'+str(i), obj=X_batch, filters=filters,)
    ds[:] = X_batch
    print(ds)
    f.close()

    if not x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed':  # do NOT write for TEST DATA
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
    x = h5f.get_node('/x_data'+str(1792)).read()  # get a specific chunk of data
    print(x)
    print('X SHAPE AFTER', np.array(x, dtype=object).shape)


# Read y batches from file (pre-processed)
with tables.File(y_filename+'.hdf', 'r') as h5f:
    y = h5f.get_node('/y_data'+str(1792)).read()  # get a specific chunk of data
    print(y)
    print('y SHAPE AFTER', np.array(y, dtype=object).shape)
