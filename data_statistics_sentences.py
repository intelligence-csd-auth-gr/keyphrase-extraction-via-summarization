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
file = 'data/kp20k_training.json'

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

# Split text to sentences
data['abstract'] = data['abstract'].apply(sent_tokenize)
print(data['abstract'][0])

# Split each row containing list of sentences to a sentence per row (one sentence is considered as one document)
data = data.explode('abstract')
print(data)

# reset index as explode results in multiple rows having the same index
data.reset_index(inplace=True)


# ======================================================================================================================
# Remove punctuation
# ======================================================================================================================

# keep punctuation that might be useful                     -  !"#$%&'()*+,./:;<=>?@[\]^_`{|}~
keep_punctuation = ['<', '=', '>', '^', '{', '|', '}', '/', '%', '$', '*', '&']
punctuation = [punct for punct in string.punctuation if punct not in keep_punctuation]
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
print('punctuation abstract finish')
# print(data)


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

# TOTAL SENTENCES: 4.189.874

above_40 = 0
for word_count in test:
    if word_count > 40:
        above_40 += 1
# above 40 = 169649
print("number of documents that will be cropped - above 40 words: ", above_40)

above_45 = 0
for word_count in test:
    if word_count > 45:
        above_45 += 1
# above 45 = 92604
print("number of documents that will be cropped - above 45 words: ", above_45)

above_50 = 0
for word_count in test:
    if word_count > 50:
        above_50 += 1
# above 50 = 52904
print("number of documents that will be cropped - above 50 words: ", above_50)

above_60 = 0
for word_count in test:
    if word_count > 60:
        above_60 += 1
# above 60 = 20400
print("number of documents that will be cropped - above 60 words: ", above_60)

above_70 = 0
for word_count in test:
    if word_count > 70:
        above_70 += 1
# above 70 = 9483
print("number of documents that will be cropped - above 70 words: ", above_70)


size = 10
print(size)

# kde + histogram
sns.distplot(test, hist=True, kde=True,
             bins=size, color='darkblue',
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 4})

# Add labels
plt.title('Density plot and Histogram of number of words of abstract sentences')
plt.xlabel('Count of sentences/documents')
plt.ylabel('Number of words')
plt.show()


# histogram
plt.hist(test, color='blue', edgecolor='black', bins=size)
# Add labels
plt.title('Histogram of number of words of abstract sentences')
plt.xlabel('Number of words')
plt.ylabel('Count of sentences/documents')
plt.show()


# ======================================================================================================================
# Find the maximum length of abstract in the whole dataset
# ======================================================================================================================

print("Maximum length of sentences of abstract in the whole dataset", max(data['abstract'].apply(len)))
