import re
import json
import string
import numpy as np
import pandas as pd
import plotly.offline as py
from pandas import json_normalize
import plotly.graph_objects as go
from nltk.tokenize import sent_tokenize, word_tokenize


pd.set_option('display.max_columns', None)

# ======================================================================================================================
# Read data file
# ======================================================================================================================

# reading the JSON data using json.load()
file = '../data/kp20k_training.json'

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

# print(data)

# get a sample of the whole dataset (for development ONLY)
'''
data = data.sample(n=1000, random_state=42)
data.reset_index(inplace=True)  # NEEDED if sample is enabled in order to use enumerate in the for loop below
'''

'''
DATA IS ALREADY CLEAN

# Clean missing data
data = data.dropna()
data.reset_index(drop=True, inplace=True)  # needed when we sample the data in order to join dataframes

print(data)
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
    # print('title_abstract_mainBody', title_abstract)

    data['abstract'].iat[index] = title_abstract


# ======================================================================================================================
# Split text to sentences so that each sentence corresponds to one document
# ======================================================================================================================

# remove period (.) from acronyms and replace e.g., i.e., etc. to avoid noise for sentence boundary detection
data['abstract'] = data['abstract'].apply(lambda text: re.sub(r'(?<!\w)([A-Z])\.', r'\1', text.replace('e.g.', 'eg')))
data['abstract'] = data['abstract'].apply(lambda text: text.replace('i.e.', 'ie'))
data['abstract'] = data['abstract'].apply(lambda text: text.replace('etc.', 'etc'))

# Split text to sentences
data['abstract'] = data['abstract'].apply(sent_tokenize)
print(data['abstract'][0])

# Split each row containing list of sentences to a sentence per row (one sentence is considered as one document)
data = data.explode('abstract')
print(data)

# reset index as explode results in multiple rows having the same index
data.reset_index(drop=True, inplace=True)


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


# remove rows with empty abstracts - empty sentences
data = data[data['abstract'].str.strip().astype(bool)]
#data = data[data['fulltext'].str.contains(' ')]  # remove single words
# reset index as explode results in multiple rows having the same index
data.reset_index(drop=True, inplace=True)
print('AFTER CLEANING', data)


# remove empty keyphrases
print('LEN BEFORE', len(data))
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
# remove rows with empty keyphrases
data = data[data['keyword'].map(len) > 0]
print('LEN AFTER', len(data))


# ======================================================================================================================
# Tokenize each sentence
# ======================================================================================================================

def tokenize_stem_lowercase(text):
    '''
    Toekenize, stem and convert to lower case the text of documents
    :param text: text of a specific document
    :return: formatted text
    '''
    formatted_text = []
    words = word_tokenize(text)  # tokenize document text
    for word_token in words:  # get words of all keyphrases in a single list
        # formatted_text.append(Stemmer('porter').stem(word_token.lower()))
        formatted_text.append(word_token.lower())
    return formatted_text


# tokenize text
data['abstract'] = data['abstract'].apply(tokenize_stem_lowercase)
print(data['abstract'])
print('tokenization abstract finish')


# ======================================================================================================================
# Histogram | Density plot of number of words per abstract/document
# ======================================================================================================================

import seaborn as sns
import matplotlib.pyplot as plt
test = data['abstract'].apply(len)
print(test)

# TOTAL SENTENCES: 4.136.306

above_30 = 0
for word_count in test:
    if word_count > 30:
        above_30 += 1
# above 300 = 641429
print("number of documents that will be cropped - above 30 words: ", above_30)

above_40 = 0
for word_count in test:
    if word_count > 40:
        above_40 += 1
# above 40 = 188384
print("number of documents that will be cropped - above 40 words: ", above_40)

above_45 = 0
for word_count in test:
    if word_count > 45:
        above_45 += 1
# above 45 = 103694
print("number of documents that will be cropped - above 45 words: ", above_45)

above_50 = 0
for word_count in test:
    if word_count > 50:
        above_50 += 1
# above 50 = 58846
print("number of documents that will be cropped - above 50 words: ", above_50)

above_60 = 0
for word_count in test:
    if word_count > 60:
        above_60 += 1
# above 60 = 22317
print("number of documents that will be cropped - above 60 words: ", above_60)

above_70 = 0
for word_count in test:
    if word_count > 70:
        above_70 += 1
# above 70 = 10024
print("number of documents that will be cropped - above 70 words: ", above_70)


size = 100
print(size)

# kde + histogram
sns.distplot(test, hist=True, kde=True,
             bins=size, color='darkblue',
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 4})

# Add labels
plt.title('Density plot and Histogram of number of words of title & abstract sentences')
plt.xlabel('Count of sentences/documents')
plt.ylabel('Number of words')
plt.show()


# histogram
plt.hist(test, color='blue', edgecolor='black', bins=size)
# Add labels
plt.title('Histogram of number of words of title & abstract sentences')
plt.xlabel('Number of words')
plt.ylabel('Count of sentences/documents')
plt.show()


# ======================================================================================================================
# Find the maximum length of abstract in the whole dataset
# ======================================================================================================================

print("Maximum length of sentences of abstract in the whole dataset", max(data['abstract'].apply(len)))
# Maximum length of sentences of abstract in the whole dataset 347
