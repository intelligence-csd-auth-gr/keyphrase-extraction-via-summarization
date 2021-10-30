import re
import json
import pickle
import string
import tables
import numpy as np
import pandas as pd
from string import digits
from pandas import json_normalize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer


from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)

# ======================================================================================================================
# Set batch size and file names in which pre-processed data will be saved
# ======================================================================================================================

# reading the initial JSON data using json.load()
file = '..\\ACM.json'  # TEST data to evaluate the final model


# Define the file paths and names to save TEST data to evaluate the final model
x_filename = '..\\..\\preprocessed_data\\paragraph_fulltext\\x_ACM_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
y_filename = '..\\..\\preprocessed_data\\paragraph_fulltext\\y_ACM_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'


# Define the file paths and names to save preprocessed TEXT of ABSTRACT and KEYPHRASES
x_text_filename = '..\\..\\preprocessed_data\\paragraph_fulltext\\x_ACM_PARAGRAPH_FULLTEXT_preprocessed_TEXT'  # save preprocessed text for TEST data
y_text_filename = '..\\..\\preprocessed_data\\paragraph_fulltext\\y_ACM_PARAGRAPH_FULLTEXT_preprocessed_TEXT'  # save preprocessed keyphrases for TEST data


# Define the number of lines to read
batch_size = 64  # 1024  # 10000
# max length of sequences - used for padding (count when the text is tokenized)
max_len = 400  # Used to match the data dimensions for both TRAIN and TEST data


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
# Isolate the title, abstract and the main body (+ remove section identifiers and '\n')
# ======================================================================================================================

# tokenize key-phrases and keep them categorized by document
for index, fulltext in enumerate(data['fulltext']):
    # extract the title
    start_title = fulltext.find("--T\n") + len("--T\n")  # skip the special characters '--T\n'
    end_title = fulltext.find("--A\n")
    title = fulltext[start_title:end_title]
    # print('title', title)

    # extract the abstract
    start_abstract = fulltext.find("--A\n") + len("--A\n")  # skip the special characters '--A\n'
    end_abstract = fulltext.find("--B\n")
    abstract = fulltext[start_abstract:end_abstract]
    # print('abstract', abstract)

    # extract the fulltext
    start_fulltext = fulltext.find("--B\n") + len("--B\n")  # skip the special characters '--B\n'
    end_fulltext = fulltext.find("--R\n")  # do not include references
    main_body = fulltext[start_fulltext:end_fulltext]

    '''
    # extract the references
    start_fulltext = fulltext.find("--R\n") + len("--R\n")
    end_fulltext = fulltext.find("--TR")
    references = fulltext[start_fulltext:end_fulltext]
    '''

    title_abstract_mainBody = title + ' ' + abstract + ' ' + main_body
    # remove '\n'
    title_abstract_mainBody = title_abstract_mainBody.replace('\n', ' ')
    # print('title + abstract', title_abstract)

    data['fulltext'].iat[index] = title_abstract_mainBody


# rename column "fulltext" to "abstract" for uniformity between datasets
data.rename(columns={"fulltext": "abstract"}, inplace=True)


# ======================================================================================================================
# Split text to paragraphs so that each sentence corresponds to one document
# ======================================================================================================================

def splitByParagraphs(doc):
    """
    split full-text documents to paragraphs by unifying sentences into paragraphs until the total length of the paragraph reaches 163 words
    :param doc: document split into sentences
    :return: document split into paragraphs
    """
    # split into sentences, then combine sentences till the length reaches 220 (which is the avg length
    # of the documents with length more than the avg for train abstracts). With 220 as a max number, we guarantee that
    # all of the paragraphs will have length below 220 and will be closer to the 163 avg
    # that way sentences are not split in the middle, retaining their semantics, could not split using the '\n\n' as
    # a separator because it was not used as a paragraph identifier in the datasets
    current_paragraph_length = 0
    paragraph = ''
    list_of_paragraphs = []
    for idx, sentence in enumerate(doc):
        current_paragraph_length += len(word_tokenize(sentence))  # tokenize sentences to count its words

        if current_paragraph_length <= max_len:
            paragraph += sentence + ' '

            if (idx + 2) > len(doc):  # if this is the last sentence of the document (+2 because counting from 0 and we want the next element)
                # save the completed paragraph
                list_of_paragraphs.append(paragraph.strip())
        else:  # length exceeds 220
            if (idx + 2) > len(doc):  # if this is the last sentence of the document (+2 because counting starts from 0 and we want the next element as well)
                #paragraph += sentence + ' '  # add the last sentence if though the length of this paragraph will exceed the boundary

                # save the completed paragraph
                list_of_paragraphs.append(paragraph.strip())

                paragraph = sentence  # save the remained sentence as a paragraph

                # save the final-sentence paragraph
                list_of_paragraphs.append(paragraph.strip())

                if len(word_tokenize(paragraph.strip())) > max_len:  # find the paragraphs surpassing the boundary
                    print('paragraph/sentence is surpassing length boundary', len(word_tokenize(paragraph.strip())))
            else:
                # save the completed paragraph
                list_of_paragraphs.append(paragraph.strip())

                # reset the length count of the current paragraph to the length of the current first added sentence
                current_paragraph_length = len(word_tokenize(sentence))  # tokenize sentences to count its words

                # reset the current paragraph to a new paragraph
                paragraph = sentence + ' '

    return list_of_paragraphs


# remove period (.) from acronyms and replace e.g., i.e., etc. to avoid noise for sentence boundary detection
data['abstract'] = data['abstract'].apply(lambda text: re.sub(r'(?<!\w)([A-Z])\.', r'\1', text.replace('e.g.', 'eg')))
data['abstract'] = data['abstract'].apply(lambda text: text.replace('i.e.', 'ie'))
data['abstract'] = data['abstract'].apply(lambda text: text.replace('etc.', 'etc'))

# Split text to sentences
data['abstract'] = data['abstract'].apply(sent_tokenize)
print(data['abstract'][0])


# unify sentences into paragraphs with targeted length 220 (avg length of train data abstracts)
data['abstract'] = data['abstract'].apply(splitByParagraphs)
print(data['abstract'][0])
for para in data['abstract'][0]:
    print(len(word_tokenize(para)))



# Split each row containing list of paragraphs to a paragraphs per row (one sentence is considered as one document)
data = data.explode('abstract')
print(data)


# save the index after explode in order to unify test/pred keyphrases from sentences back into original documents
data['assemble_documents_index'] = data.index

# reset index as explode results in multiple rows having the same index
data.reset_index(drop=True, inplace=True)  # drop=True: drops column level_0 generated by reset_index

'''
print(data)
import sys
sys.exit()  # terminate program to save the number of total documents-rows-sentences
'''

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
# Remove punctuation, non-ascii characters (with whitespace) + digits (from ABSTRACT)
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


# keep punctuation that might be useful                     -  !"#$%&'()*+,./:;<=>?@[\]^_`{|}~
#keep_punctuation = ['<', '=', '>', '^', '{', '|', '}', '/', '%', '$', '*', '&']
#punctuation = [punct for punct in string.punctuation if punct not in keep_punctuation]
punctuation = string.punctuation  # + '\t' + '\n'
#punctuation = punctuation.replace("'", '')  # do not delete '
table = str.maketrans(punctuation, ' '*len(punctuation))  # OR {key: None for key in string.punctuation}
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


# REMOVE
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# <=>\^{|}          /
# % $ -             & *

# remove punctuation
data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)
print(data['keyword'])


'''
# Remove digits
remove_digits = str.maketrans('', '', digits)  # remove digits
data['abstract'] = data['abstract'].apply(lambda text: " ".join(text.translate(remove_digits).split()))  # remove spaces
print(data['abstract'])
print('remove digits - abstract finish')
'''


# Replace the pure digit terms with DIGIT_REPL
data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
# Replace the combination of characters and digits with WORD_DIGIT_REPL
#data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('.*\d+', token) else 'WORD_DIGIT_REPL' for token in text.split()]))  # remove spaces
print(data['abstract'][3])   # 3D  ->  WORD_DIGIT_REPL
print('convert digits - abstract finish')

print(' CLEANING', data)
# remove rows with empty abstracts - empty sentences
data = data[data['abstract'].str.strip().astype(bool)]
# reset index as explode results in multiple rows having the same index
print('AFTER CLEANING', data)
data.reset_index(drop=True, inplace=True)
print('AFTER CLEANING', data)


# remove empty keyphrases
print('LEN BEFORE', len(data))
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
# remove rows with empty keyphrases
data = data[data['keyword'].map(len) > 0]
print('LEN AFTER', len(data))


# ======================================================================================================================
# Clean and translate text data to numbers (convert words to numbers)
# ======================================================================================================================

# load tokenizer
with open('..\\..\\train_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# CHECK IF LOADED TOKENIZER CONVERTS VALIDATION/TEST DATA TO LOWER CASE
X = tokenizer.texts_to_sequences(data['abstract'])
# print('sequences',X)

word_index = tokenizer.word_index
# print('word_index', word_index)


# ======================================================================================================================
# Tokenize each sentence + remove digits (from KEYPHRASES)
# ======================================================================================================================

def tokenize_lowercase(text):
    '''
    Toekenize, stem and convert to lower case the text of documents
    :param text: text of a specific document
    :return: formatted text
    '''
    formatted_text = []
    words = word_tokenize(text)  # tokenize document text
    for word_token in words:  # get words of all keyphrases in a single list
        formatted_text.append(word_token.lower())
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
#    print('THESE ARE THE KEYPHRASE LIST', len(keyphrases_list), keyphrases_list)


# ======================================================================================================================
# Write pre-processed keyphrases to csv file
# ======================================================================================================================

data['abstract'].to_csv(x_text_filename, index=False)  # save the preprocessed document text
data[['keyword', 'assemble_documents_index']].to_csv(y_text_filename, index=False)  # save the preprocessed keyphrases


# ======================================================================================================================
# Give labels to each word of Abstract (fulltext) - keyword (KP) or Non-keyword (Non-KP)
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
'''
KP count:  168345 
KP WORDS count:  261284 
NON-KP count:  15598434
X SHAPE (53083, 2161)
'''


# ======================================================================================================================
# Save pre-processed X and y values in file in order to change the batch size efficiently
# ======================================================================================================================
import json
# save data
with open(x_filename+".txt", "w") as fp_x:
    json.dump(X, fp_x)
with open(y_filename+".txt", "w") as fp_y:
    json.dump(y, fp_y)

''' LOAD DATA
with open(x_filename+".txt", "r") as fp:
    X = json.load(fp)
with open(y_filename+".txt", "r") as fp:
    y = json.load(fp)
print(X)
print(y)
'''



# ======================================================================================================================
# Necessary pre-processing for Bi-LSTM (in general for neural networks)
# ======================================================================================================================

# Find the maximum length of abstract in the whole dataset
# Max length of abstract (fulltext) (2161)
# max_len = max(data['abstract'].apply(len))  # Max length of abstract
print('X SHAPE', pd.DataFrame(X).shape)  # Max length of title and abstract

# Set Abstract + Title max word size (text longer than the number will be trancated)
# max_len = 348  # Maximum length of abstract in the TRAIN data
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
    X_batch = pad_sequences(sequences=X[i:i + batch_size], padding="post", maxlen=max_len, value=0)

    print('X SHAPE AFTER', np.array(X_batch, dtype=object).shape)
#    print('y SHAPE AFTER', np.array(y_batch, dtype=object).shape)


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


    # free memory here in order to allow for bigger batches (not needed, but allows for bigger sized batches)
    X_batch = None


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
