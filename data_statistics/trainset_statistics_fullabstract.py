import re
import json
import string
import numpy as np
import pandas as pd
import plotly.offline as py
import matplotlib.pyplot as plt
from pandas import json_normalize
import plotly.graph_objects as go
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer



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


# ======================================================================================================================
# Split keyphrases list of keyphrases from string that contains all the keyphrases
# ======================================================================================================================

for index, keywords in enumerate(data['keyword']):
    data['keyword'].iat[index] = keywords.split(';')  # split keywords to separate them from one another

# print(data)

# get a sample of the whole dataset (for development ONLY)
'''
data = data.sample(n=1000, random_state=42)
data.reset_index(inplace=True)  # NEEDED if sample is enabled in order to use enumerate in the for loop below
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

data['title'] = data['title'].apply(replace_contractions)

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
data['title'] = data['title'].apply(remove_brackets_and_contents)


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
data['title'] = data['title'].apply(remove_references)



punctuation = string.punctuation
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
data['title'] = data['title'].apply(remove_punct_and_non_ascii)
data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)
print(data['keyword'])

'''
# Remove digits
remove_digits = str.maketrans('', '', digits)  # remove digits
data['abstract'] = data['abstract'].apply(lambda text: " ".join(text.translate(remove_digits).split()))  # remove spaces
print(data['abstract'])
print('remove digits - abstract finish')
'''

# Replace the pure digit terms with DIG IT_REPL and REMOVE REDUNDANT WHITESPACES
data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
data['title'] = data['title'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
# Replace the combination of characters and digits with WORD_DIGIT_REPL
# data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('.*\d+', token) else 'WORD_DIGIT_REPL' for token in text.split()]))  # remove spaces
print(data['abstract'][3])  # 3D  ->  WORD_DIGIT_REPL
print('convert digits - abstract finish')

# remove rows with empty abstracts - empty sentences
data = data[data['abstract'].str.strip().astype(bool)]
data = data[data['title'].str.strip().astype(bool)]
# data = data[data['fulltext'].str.contains(' ')]  # remove single words
# reset index as explode results in multiple rows having the same index
data.reset_index(drop=True, inplace=True)
print('AFTER CLEANING', data)


# remove empty keyphrases
print('LEN BEFORE', len(data))
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
# remove rows with empty sets of keyphrases
data = data[data['keyword'].map(len) > 0]
print('LEN AFTER', len(data))


# ======================================================================================================================
# Tokenize each sentence + remove digits (from KEYPHRASES)
# ======================================================================================================================

def tokenize_lowercase(text):
    """
    Toekenize, stem and convert to lower case the text of documents
    :param text: text of a specific document
    :return: formatted text
    """
    words = word_tokenize(text)  # tokenize document text
    # get words of all keyphrases in a single list
    formatted_tok_text = [Stemmer('porter').stem(word_token.lower()) for word_token in words]  # DO NOT STEM TEXT WORDS TO TRAIN THE CLASSIFIER
    formatted_text = ' '.join(formatted_tok_text)
    return formatted_text


# tokenize text
data['abstract'] = data['abstract'].apply(tokenize_lowercase)
data['title'] = data['title'].apply(tokenize_lowercase)
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
            # tokens = [tok if not re.match('.*\d+', tok) else 'WORD_DIGIT_REPL' for tok in tokens]
            tokens = [Stemmer('porter').stem(keyword.lower()) for keyword in tokens]  # stem + lower case
            tokens = ' '.join(tokens)
            keyphrases_list.append(tokens)

    data['keyword'].iat[index] = keyphrases_list




# ======================================================================================================================
# Count logistics
# ======================================================================================================================

keywords_in_title = 0  # the count of keywords in title
keywords_in_abstract = 0  # the count of keywords in abstract
keywords_in_title_abstract = 0  # the count of keywords that are either in title or abstract
keywords_in_title_NOT_abstract = 0  # the count of keywords that are in title BUT NOT in abstract
total_keywords = 0  # the count of all keywords
for index, keywords in enumerate(data['keyword']):

    total_keywords += len(keywords)
    # print('total_keywords', len(test))
    # print('total_keywords', test)

    for keyword in keywords:
        # check if keyword exists on title
        if keyword in data['title'][index]:
            keywords_in_title += 1
            # print(keyword)
            # print(data['title'][index])

        # check if keyword exists on abstract
        if keyword in data['abstract'][index]:
            keywords_in_abstract += 1
            # print(keyword)
            # print(data['abstract'][index])

        # check if keyword exists either on title or abstract
        if keyword in data['title'][index] or keyword in data['abstract'][index]:
            keywords_in_title_abstract += 1
            # print(keyword)

        if keyword in data['title'][index] and keyword not in data['abstract'][index]:
            keywords_in_title_NOT_abstract += 1
            # print(keyword)

print('title: ', keywords_in_title)
print('abstract: ', keywords_in_abstract)
print('either in title or abstract: ', keywords_in_title_abstract)
print('in title, NOT in abstract: ', keywords_in_title_NOT_abstract)
print('total keyphrases: ', total_keywords)

print('count of keywords in title: ', keywords_in_title / total_keywords)
print('count of keywords in abstract: ', keywords_in_abstract / total_keywords)
print('count of keywords that are either in title or abstract: ', keywords_in_title_abstract / total_keywords)
print('count of keywords that are in title and NOT in abstract: ', keywords_in_title_NOT_abstract / total_keywords)


# ======================================================================================================================
# Visualize the counts of keywords in title, abstract and combinations
# ======================================================================================================================

# Barplot of percentages of keywords in title, abstract and combinations
key_title = 100 * np.round(keywords_in_title / total_keywords, 4)
key_abstract = 100 * np.round(keywords_in_abstract / total_keywords, 4)
key_title_abstract = 100 * np.round(keywords_in_title_abstract / total_keywords, 4)
key_title_not_abstract = 100 * np.round(keywords_in_title_NOT_abstract / total_keywords, 4)

count_names = ['Keyphrases in title', 'Keyphrases in abstract', 'Keyphrases either in title or abstract',
               'Keyphrases in title and NOT in abstract']
count_scores = [key_title, key_abstract, key_title_abstract, key_title_not_abstract]
colors = ['cyan', 'crimson', 'coral', 'cadetblue']
popup_text = ['Count of keyphrases in title: {}'.format(keywords_in_title),
              'Count of keyphrases in abstract: {}'.format(keywords_in_abstract),
              'Count of keyphrases in title-abstract: {}'.format(keywords_in_title_abstract),
              'Count of keyphrases in title not abstract: {}'.format(keywords_in_title_NOT_abstract)]

fig = go.Figure(data=[go.Bar(x=count_names, y=count_scores, text=popup_text, marker_color=colors)],
                layout=go.Layout(
                    yaxis=dict(range=[0, 100],  # sets the range of yaxis
                               constrain="domain")  # meanwhile compresses the yaxis by decreasing its "domain"
                )
                )

fig.update_layout(title_text='Percentages of keywords in title, abstract and combinations - Total count of gold keyphrases: {}'.format(total_keywords), title_x=0.5)
fig.update_yaxes(ticksuffix="%")

py.plot(fig, filename='..//schemas//preprocess_barplot_keyphrase_counting.html')

'''
TRAIN DATA -> DO FOR TEST DATA AS WELL

avg of abstract below avg: 113 
avg: 163
avg of abstract above avg: 220
max length: 3023

UNIQUE WORDS (does not count each time they appear in a document, just once for each document)
title:  796666
abstract:  1652657
either in title or abstract:  1774628
in title, NOT in abstract:  121971
total keyphrases:  2801398
count of keywords in title:  0.28438158376639094
count of keywords in abstract:  0.5899400941958265
count of keywords that are either in title or abstract:  0.6334794270574906
count of keywords that are in title and NOT in abstract:  0.043539332861664067
'''


colors_list = ['#5cb85c', '#5bc0de', '#FFA500', '#d9534f']

# kp_percentages = [28.44, 58.99, 63.35, 4.35]  # (actual numbers - not dummies)
# kp_counts = [796666, 1652657, 1774628, 121971]  # (actual numbers - not dummies)
kp_percentages = [key_title, key_abstract, key_title_abstract, key_title_not_abstract]
kp_counts = [keywords_in_title, keywords_in_abstract, keywords_in_title_abstract, keywords_in_title_NOT_abstract]
labels = ['KPs in title', 'KPs in abstract', 'KPs in title & \n abstract', 'KPs in title & \n not in abstract']


x = [25, 50, 75, 100]  # the label locations
width = 10  # the width of the bars

plt.figure(figsize=(8, 6), dpi=80)
rects = plt.bar(x, kp_percentages, width, color=colors_list)

# add percentage symbol behind the values of the y axis
plt.gca().set_yticklabels(['{:.0f}%'.format(y_axis) for y_axis in plt.gca().get_yticks()])

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.title("Keyphrase coverage percentage of title, abstract and combinations \n Total count of gold keyphrases: {}".format(total_keywords), pad=10)
plt.xlabel('Information source', labelpad=10)
plt.ylabel('Percentage of keyphrase coverage', labelpad=10)
plt.xticks(ticks=x, labels=labels, fontsize=9)
plt.xlim([0, 125])
plt.ylim([0, 100])


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index, rect in enumerate(rects):
        height = rect.get_height()
        plt.annotate('{:.2f}% \n {} keyphrases'.format(height, kp_counts[index]),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',  fontsize=9)


autolabel(rects)

plt.tight_layout()

plt.show()

# plt.savefig('trainset_statistics.png')  # save the plot of model's loss per epoch to file



# ======================================================================================================================
# Pre-processing
# ======================================================================================================================

# tokenize text
data['title'] = data['title'].apply(word_tokenize)
print('title finish')
data['abstract'] = data['abstract'].apply(word_tokenize)
print('abstract finish')
'''
# tokenize key-phrases and keep them categorized by document
for index, list_of_keyphrases in enumerate(data['keyword']):
    test = []
    for keyphrase in list_of_keyphrases:  # get words of all keyphrases in a single list
        testing = word_tokenize(keyphrase)
        for keyword in testing:
            test.append(keyword)
    test = set(test)  # keep only the unique words
    data['keyword'].iat[index] = test
'''


# ======================================================================================================================
# Histogram | Density plot of number of words per abstract/document
# ======================================================================================================================

import seaborn as sns
from operator import add
import matplotlib.pyplot as plt
abstract_length = data['abstract'].apply(len)
title_length = data['title'].apply(len)
print(abstract_length)
# calculate the lengths of title + abstract
doc_length = list(map(add, abstract_length, title_length))  # element wise addition of the title and abstract lengths
print(doc_length)

above_300 = 0
for word_count in doc_length:
    if word_count > 300:
        above_300 += 1
# above 300 = 14914
print("number of documents that will be cropped - above 300 words: ", above_300)

above_400 = 0
for word_count in doc_length:
    if word_count > 400:
        above_400 += 1
# above 400 = 2204
print("number of documents that will be cropped - above 400 words: ", above_400)

above_500 = 0
for word_count in doc_length:
    if word_count > 500:
        above_500 += 1
# above 500 = 653
print("number of documents that will be cropped - above 500 words: ", above_500)

above_600 = 0
for word_count in doc_length:
    if word_count > 600:
        above_600 += 1
# above 600 = 329
print("number of documents that will be cropped - above 600 words: ", above_600)

above_700 = 0
for word_count in doc_length:
    if word_count > 700:
        above_700 += 1
# above 700 = 182
print("number of documents that will be cropped - above 700 words: ", above_700)

above_800 = 0
for word_count in doc_length:
    if word_count > 800:
        above_800 += 1
# above 800 = 123
print("number of documents that will be cropped - above 800 words: ", above_800)


size = 100
print(size)

# kde + histogram
sns.distplot(doc_length, hist=True, kde=True,
             bins=size, color='darkblue',
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 4})

# Add labels
plt.title('Density plot and Histogram of number of words of abstract')
plt.xlabel('Count of documents')
plt.ylabel('Number of words')
plt.show()


# histogram
plt.hist(doc_length, color='blue', edgecolor='black', bins=size)
# Add labels
plt.title('Histogram of number of words of abstract')
plt.xlabel('Number of words')
plt.ylabel('Count of documents')
plt.show()


import plotly.figure_factory as ff
group_labels = ['distplot']  # name of the dataset
fig = ff.create_distplot([doc_length], group_labels)
py.plot(fig, filename='../schemas/preprocess_distplot_word_count.html')


# ======================================================================================================================
# Find the maximum length of abstract in the whole dataset
# ======================================================================================================================

print("Maximum length of abstract in the whole dataset", max(data['abstract'].apply(len))) # 3011
print("Maximum length of title in the whole dataset", max(data['title'].apply(len)))
# Maximum length of abstract in the whole dataset 2495
# Maximum length of title in the whole dataset 90
