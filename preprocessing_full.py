import re
import sys
import json
import pickle
import string
import tables
import numpy as np
import pandas as pd
from string import digits
from pandas import json_normalize
from numpy import savez_compressed
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.snowball import SnowballStemmer as Stemmer


from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)


# ======================================================================================================================
# Argument parsing
# ======================================================================================================================

parser = ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="choose which type of data to create (options are: train, validation or test)")

args = parser.parse_args()


# ======================================================================================================================
# Set batch size and file names in which pre-processed data will be saved
# ======================================================================================================================

if args.mode == 'train':
    # reading the initial JSON data using json.load()
    file = 'data\\kp20k_training.json'  # TRAIN data

    # Define the file paths and names to save TRAIN data
    x_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed'
elif args.mode == 'validation':
    # reading the initial JSON data using json.load()
    file = 'data\\kp20k_validation.json'  # VALIDATION data to tune model parameters

    # Define the file paths and names to save VALIDATION data to tune model parameters
    x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'
elif args.mode == 'test':
    # reading the initial JSON data using json.load()
    file = 'data\\kp20k_testing.json'  # TEST data to evaluate the final model

    # Define the file paths and names to save TEST data to evaluate the final model
    x_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TEST_data_preprocessed'
else:
    print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
    sys.exit()

# save preprosessed text for TEST data - use for EVALUATION (exact/partial matching)
x_text_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT'  # save preprosessed text for TEST data
y_text_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT'  # save preprosessed keyphrases for TEST data


# Define the number of lines to read
batch_size = 64  # 1024  # 10000
max_len = 400  # Used to match the data dimensions for both TRAIN and TEST data


# ======================================================================================================================
# Read data
# ======================================================================================================================

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)



# ======================================================================================================================
# Split keyphrases list of keyphrases from string that contains all the keyphrases
# ======================================================================================================================

for index, keywords in enumerate(data['keyword']):
    data['keyword'].iat[index] = keywords.split(';')  # split keywords to separate them from one another


# ======================================================================================================================
# Combine the title and abstract (+ remove '\n')
# ======================================================================================================================

for index, abstract in enumerate(data['abstract']):
    title_abstract =  data['title'][index] + '. ' + abstract  # combine title + abstract
    # remove '\n'
    title_abstract = title_abstract.replace('\n', ' ')

    data['abstract'].iat[index] = title_abstract


# ======================================================================================================================
# Remove Contractions (pre-processing)
# ======================================================================================================================

def get_contractions():
    contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                        "could've": "could have", "couldn't": "could not", "didn't": "did not",
                        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                        "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                        "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                        "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                        "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                        "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                        "she'll've": "she will have", "she's": "she is", "should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                        "so's": "so as", "this's": "this is", "that'd": "that would",
                        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                        "there'd've": "there would have", "there's": "there is", "here's": "here is",
                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                        "they'll've": "they will have", "they're": "they are", "they've": "they have",
                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                        "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                        "when've": "when have", "where'd": "where did", "where's": "where is",
                        "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                        "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have", "nor": "not", "'s": "s", "s'": "s"}

    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

def replace_contractions(text):
    contractions, contractions_re = get_contractions()

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


# substitute contractions with full words
print('BEFORE contractions data[abstract]', data['abstract'])
data['abstract'] = data['abstract'].apply(replace_contractions)
print('AFTER contractions data[abstract]', data['abstract'])

print('BEFORE contractions data[keyword]', data['keyword'])
data['keyword'] = data['keyword'].apply(lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])
print('AFTER contractions data[keyword]', data['keyword'])


# ======================================================================================================================
# Remove punctuation (with whitespace) + digits (from ABSTRACT)
# ======================================================================================================================

# remove parenthesis, brackets and their contents
def remove_brackets_and_contents(doc):
    """
    remove parenthesis, brackets and their contents
    :param doc: initial text document
    :return: text document without parenthesis, brackets and their contents
    """
    ret = ''
    skip1c = 0
    # skip2c = 0
    for i in doc:
        if i == '[':
            skip1c += 1
        # elif i == '(':
            # skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        # elif i == ')'and skip2c > 0:
            # skip2c -= 1
        elif skip1c == 0:  # and skip2c == 0:
            ret += i
    return ret


# remove parenthesis, brackets and their contents
data['abstract'] = data['abstract'].apply(remove_brackets_and_contents)


# delete newline and tab characters
newLine_tabs = '\t' + '\n'
newLine_tabs_table = str.maketrans(newLine_tabs, ' ' * len(newLine_tabs))
print(newLine_tabs, 'newLine_tabs LEN:', len(newLine_tabs))


# remove references of publications (in document text)
def remove_references(doc):
    """
    remove references of publications (in document text)
    :param doc: initial text document
    :return: text document without references
    """
    # delete newline and tab characters
    clear_doc = doc.translate(newLine_tabs_table)

    # remove all references of type "Author, J. et al., 2014"
    clear_doc = re.sub(r'[A-Z][a-z]+,\s[A-Z][a-z]*\. et al.,\s\d{4}', "REFPUBL", clear_doc)

    # remove all references of type "Author et al. 1990"
    clear_doc = re.sub("[A-Z][a-z]+ et al. [0-9]{4}", "REFPUBL", clear_doc)

    # remove all references of type "Author et al."
    clear_doc = re.sub("[A-Z][a-z]+ et al.", "REFPUBL", clear_doc)

    return clear_doc


# remove references of publications (in document text)
data['abstract'] = data['abstract'].apply(remove_references)


punctuation = string.punctuation
table = str.maketrans(punctuation, ' '*len(punctuation))
print(punctuation, 'LEN:', len(punctuation))


# remove special characters, non-ascii characters, all single letter except from 'a' and 'A', and, punctuations with whitespace
def remove_punct_and_non_ascii(text):
    clean_text = text.translate(table)
    clean_text = clean_text.encode("ascii", "ignore").decode()  # remove non-ascii characters
    # remove all single letter except from 'a' and 'A'
    clean_text = re.sub(r"\b[b-zB-Z]\b", "", clean_text)
    return clean_text


def keyword_remove_punct_and_non_ascii(text):
    # remove all single letter except from 'a' and 'A'
    clean_text = [re.sub(r"\b[b-zB-Z]\b", "", keyw.translate(table).encode("ascii", "ignore").decode()) for keyw in text]  # remove non-ascii characters
    return clean_text


# remove punctuation
data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)


# Replace the pure digit terms with DIG IT_REPL and REMOVE REDUNDANT WHITESPACES
data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
# Replace the combination of characters and digits with WORD_DIGIT_REPL
#data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('.*\d+', token) else 'WORD_DIGIT_REPL' for token in text.split()]))  # remove spaces


# remove rows with empty abstracts - empty sentences
data = data[data['abstract'].str.strip().astype(bool)]
# reset index as explode results in multiple rows having the same index
data.reset_index(drop=True, inplace=True)


# remove empty keyphrases
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
# remove rows with empty sets of keyphrases
data = data[data['keyword'].map(len) > 0]


# ======================================================================================================================
# Clean and translate text data to numbers (convert words to numbers)
# ======================================================================================================================

# Only for the train files, create the GloVe matrix, that will be used as weights in the embedding layer
if x_filename == 'data\\preprocessed_data\\x_TRAIN_data_preprocessed':  # if the data are the TEST DATA

    # oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls
    tokenizer = Tokenizer(filters='', lower=True, oov_token='<UKN>')
    tokenizer.fit_on_texts(data['abstract'])

    # convert text to sequence of numbers
    X = tokenizer.texts_to_sequences(data['abstract'])

    # get the word-index pairs
    word_index = tokenizer.word_index

    # save tokenizer
    with open('data\\train_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:  # for validation and test sets

    # load tokenizer
    with open('data\\train_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    X = tokenizer.texts_to_sequences(data['abstract'])

    word_index = tokenizer.word_index


# ======================================================================================================================
# Tokenize each sentence + remove digits (from KEYPHRASES)
# ======================================================================================================================

def tokenize_lowercase(text):
    """
    Toekenize, stem and convert to lower case the text of documents
    :param text: text of a specific document
    :return: formatted text
    """
    formatted_text = []
    words = word_tokenize(text)  # tokenize document text
    for word_token in words:  # get words of all keyphrases in a single list
        formatted_text.append(word_token.lower())  # DO NOT STEM TEXT WORDS TO TRAIN THE CLASSIFIER
    return formatted_text


# tokenize text
data['abstract'] = data['abstract'].apply(tokenize_lowercase)
print(data['abstract'])
print('tokenization - abstract finish')


# stem, tokenize and lower case keyphrases and keep them categorized by document
for index, list_of_keyphrases in enumerate(data['keyword']):
    keyphrases_list = []
    for keyphrase in list_of_keyphrases:  # get words of all keyphrases in a single list
        # keyphrase = keyphrase.translate(remove_digits).strip()  # remove digits
        keyphrase = keyphrase.strip()  # remove whitespaces
        if len(keyphrase):  # check if the keyphrase is empty
            tokens = word_tokenize(keyphrase)  # tokenize
            # Replace the pure digit terms with DIGIT_REPL
            tokens = [tok if not re.match('^\d+$', tok) else 'DIGIT_REPL' for tok in tokens]
            # Replace the combination of characters and digits with WORD_DIGIT_REPL
            #tokens = [tok if not re.match('.*\d+', tok) else 'WORD_DIGIT_REPL' for tok in tokens]
            keyphrases_list.append([Stemmer('porter').stem(keyword.lower()) for keyword in tokens])  # stem + lower case
    data['keyword'].iat[index] = keyphrases_list


# ======================================================================================================================
# Write pre-processed keyphrases to csv file (if the data are the TEST DATA)
# ======================================================================================================================

if x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':  # if the data are the TEST DATA
    data['abstract'].to_csv(x_text_filename, index=False)  # save the preprocessed document text
    data['keyword'].to_csv(y_text_filename, index=False)  # save the preprocessed keyphrases


# ======================================================================================================================
# GloVe vectors
# ======================================================================================================================

# Only for the train files, create the GloVe matrix, that will be used as weights in the embedding layer
if x_filename == 'data\\preprocessed_data\\x_TRAIN_data_preprocessed':  # if the data are the TEST DATA

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

    print("Found %s word vectors." % len(glove_model))

    # get the vector of a word
    def get_glove_vec(word_vec):
        """
        Get the vector of a word
        :param word_vec: a given word
        :return: the GloVe vector of given word
        """
        
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

    import operator
    def check_coverage(vocab, glove_model):
        a = {}
        oov = {}
        k = 0
        i = 0
        for word in tqdm(vocab):
            try:
                a[word] = glove_model[word]
                k += vocab[word]
            except:

                oov[word] = vocab[word]
                i += vocab[word]
                pass

        print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
        print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
        sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

        return sorted_x


    oov = check_coverage(tokenizer.word_counts, glove_model)  # word_counts
    print('out of vocab: ', oov[:30])


# ======================================================================================================================
# Get the GloVe representation matrix that matches document words (as numeric values) to GloVe vectors
# ======================================================================================================================

    embedding_dim = 100
    num_tokens = len(word_index)
    print('Vocabulary (number of unique words):', num_tokens)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = get_glove_vec(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be equal to the median of the GloVe vector space (handled by get_glove_vec function).
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector

    print('embedding_matrix', embedding_matrix)
    print('embedding_matrix %s.' % len(embedding_matrix))



    # save the GloVe embedding matrix to file in order to use as weight for the embeddings layer of Bi-LSTM-CRF
    savez_compressed('data\\embedding_matrix.npz', embedding_matrix)


# ======================================================================================================================
# Give labels to each word of Title and Abstract - keyword (KP) or Non-keyword (Non-KP)
# ======================================================================================================================

# Convert Tag/Label to tag_index
count_KP = 0  # count the number of keyphrases (used for cost-sensitive learning in training of Bi-LSTM-CRF)
count_KP_words = 0  # count the number of key WORDS (used for cost-sensitive learning in training of Bi-LSTM-CRF)
count_NON_KP = 0  # count the number of NON keyphrases (used for cost-sensitive learning in training of Bi-LSTM-CRF)
y = []  # list of lists that contain the labels (keyword (KP), Non-keyword (Non-KP)) for each word of the abstract
for index, abstract in enumerate(tqdm(data['abstract'])):
    abstract_word_labels = [0] * len(abstract)  # keep the labels of the all the words contained in a single title and abstract text (initialize with 0s - Non-KP value)

    # add labels for words in abstract
    for i, word in enumerate(abstract):  # check the next word of keyphrase with the next word of abstract until the full keyphrase if found
        for keyphrase in data['keyword'][index]:
            if Stemmer('porter').stem(word) == keyphrase[0]:  # locate the beginning of a keyphrase in the abstract
                match_count = 1  # how many keywords of a keyphrase match to sequential words in abstract (start with 1 as the first word already matched)
                for j in range(1, len(keyphrase)):  # check the words after the first matching keyword
                    if i + j < len(abstract):  # check if the index get ouf of abstract bounds
                        if Stemmer('porter').stem(abstract[i + j]) == keyphrase[j]:  # the word is part of a keyphrase
                            match_count += 1
                        else:
                            break  # a word in the abstract does not match with the words in keyphrase
                    else:
                        break  # reached the end of the abstract (prevent out of bounds)
                if match_count == len(keyphrase):
                    for x in range(len(keyphrase)):  # keyphrase found!
                        abstract_word_labels[i + x] = 1
                    count_KP += 1
                    break  # end iteration when a keyphrase is found to avoid checking the rest keyphrases (it will convert the labels of the keyphrase in abstract to Non-KP)
                # else:  # the word/phrase is not a keyphrase
                    # increasing count_NON_KP here is wrong (increases even when KP is found, but the KP is last on the list of KPs)
                    # abstract_word_labels[i] = 0  # not needed as abstract_word_labels is initialized with 0s
            # else:  # the word is not a keyphrase
                # if not abstract_word_labels[i]:  # (NOT needed) check if the current word has been annotated as keyword to avoid changing its value on abstract_word_labels
                    # abstract_word_labels[i] = 0  # NOT needed as abstract_word_labels is initialized with 0s
                    # count_NON_KP += 1  # increasing count_NON_KP here is wrong (increases even when KP is found, but the KP is last on the list of KPs)
        if not abstract_word_labels[i]:  # count NON-KPs
            count_NON_KP += 1
    count_KP_words += abstract_word_labels.count(1)  # count KP WORDS
    # save labels for each title + abstract
    y.append(abstract_word_labels)

print('KP count: ', count_KP, '\nKP WORDS count: ', count_KP_words, '\nNON-KP count: ', count_NON_KP)



# ======================================================================================================================
# Save pre-processed X and y values in file in order to change the batch size efficiently
# ======================================================================================================================
import json
# save data
with open(x_filename+".txt", "w") as fp_x:
    json.dump(X, fp_x)
with open(y_filename+".txt", "w") as fp_y:
    json.dump(y, fp_y)


# ======================================================================================================================
# Necessary pre-processing for Bi-LSTM (in general for neural networks)
# ======================================================================================================================

# Find the maximum length of abstract in the whole dataset
# Set Abstract + Title max word size (text longer than the number will be trancated)
# max_len = 2763  # Maximum length of abstract in the TRAIN data
print("Maximum length of title and abstract in the whole dataset", max_len)

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
    if not x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':

        # Padding each sentence to have the same length - padding values are padded to the end
        # value: padding value is set to 0, because the padding value CANNOT be a keyphrase
        y_batch = pad_sequences(sequences=y[i:i + batch_size], padding="post", maxlen=max_len, value=0)


# ======================================================================================================================
# Convert y values to CATEGORICAL
# ======================================================================================================================

        # REQUIRED - transform y to categorical (each column is a value of y - like in one-hot-encoding, columns are the vocabulary)
        y_batch = [to_categorical(i, num_classes=2, dtype='int8') for i in y_batch]


# ======================================================================================================================
# Write pre-processed TRAIN data to csv file
# ======================================================================================================================

    # Set the compression level
    filters = tables.Filters(complib='blosc', complevel=5)

    # Save X batches into file
    f = tables.open_file(x_filename+'.hdf', 'a')
    ds = f.create_carray('/', 'x_data'+str(i), obj=X_batch, filters=filters)
    ds[:] = X_batch
    f.close()

    if not x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':  # do NOT write for TEST DATA
        # Save y batches into file
        f = tables.open_file(y_filename + '.hdf', 'a')
        ds = f.create_carray('/', 'y_data' + str(i), obj=y_batch, filters=filters)
        ds[:] = y_batch
        f.close()


    # free memory here in order to allow for bigger batches (not needed, but allows for bigger sized batches)
    X_batch = None
    y_batch = None


if y_filename == 'data\\preprocessed_data\\y_TEST_data_preprocessed':  # write ONLY for TEST DATA
    y_test = pd.DataFrame({'y_test_keyword': y})
    y_test['y_test_keyword'].to_csv(y_filename, index=False)  # save the preprocessed keyphrases


# ======================================================================================================================
# Read data in chunks
# ======================================================================================================================

# Read X batches from file (pre-processed)
with tables.File(x_filename+'.hdf', 'r') as h5f:
    x = h5f.get_node('/x_data'+str(1024)).read()  # get a specific chunk of data
    print(x)
    print('X SHAPE AFTER', np.array(x, dtype=object).shape)


if not x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':  # write ONLY for TEST DATA
    # Read y batches from file (pre-processed)
    with tables.File(y_filename+'.hdf', 'r') as h5f:
        y = h5f.get_node('/y_data'+str(1024)).read()  # get a specific chunk of data
        print(y)
        print('y SHAPE AFTER', np.array(y, dtype=object).shape)
