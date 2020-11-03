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
# Count logistics
# ======================================================================================================================

keywords_in_title = 0  # the count of keywords in title
keywords_in_abstract = 0  # the count of keywords in abstract
keywords_in_title_abstract = 0  # the count of keywords that are either in title or abstract
keywords_in_title_NOT_abstract = 0  # the count of keywords that are in title BUT NOT in abstract
total_keywords = 0  # the count of all keywords
for index, keywords in enumerate(data['keyword']):
    test = keywords.split(';')

    total_keywords += len(test)
    # print('total_keywords', len(test))
    # print('total_keywords', test)

    for keyword in test:
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

fig.update_layout(title_text='Percentages of keywords in title, abstract and combinations - Total count of keyphrases: {}'.format(total_keywords), title_x=0.5)
fig.update_yaxes(ticksuffix="%")

py.plot(fig, filename='schemas//barplot_keyphrase_counting.html')

'''
title:  473356
abstract:  1163330
either in title or abstract:  1280006
in title, NOT in abstract:  116676
total keywords:  2806691
count of keywords in title:  0.1686526945787762
count of keywords in abstract:  0.41448453000348096
count of keywords that are either in title or abstract:  0.45605519097043457
count of keywords that are in title and NOT in abstract:  0.04157066096695361
'''


# ======================================================================================================================
# Pre-processing
# ======================================================================================================================

# keep punctuation that might be useful                     !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
keep_punctuation = ['<', '=', '>', '^', '{', '|', '}', '/', '%', '$', '-', '*', '&']
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


# remove punctuation
data['title'] = data['title'].apply(remove_punct)

data['abstract'] = data['abstract'].apply(remove_punct)

# data['keyword'] = data['keyword'].apply(keyword_remove_punct)
# print(data)


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
import matplotlib.pyplot as plt
test = data['abstract'].apply(len)
print(test)

above_400 = 0
for word_count in test:
    if word_count > 400:
        above_400 += 1
# above 400 = 1759
print("number of documents that will be cropped - above 400 words: ", above_400)

above_500 = 0
for word_count in test:
    if word_count > 500:
        above_500 += 1
# above 500 = 557
print("number of documents that will be cropped - above 500 words: ", above_500)

above_600 = 0
for word_count in test:
    if word_count > 600:
        above_600 += 1
# above 600 = 282
print("number of documents that will be cropped - above 600 words: ", above_600)

above_700 = 0
for word_count in test:
    if word_count > 700:
        above_700 += 1
# above 700 = 172
print("number of documents that will be cropped - above 700 words: ", above_700)

above_800 = 0
for word_count in test:
    if word_count > 800:
        above_800 += 1
# above 800 = 111
print("number of documents that will be cropped - above 800 words: ", above_800)


size = 100
print(size)

# kde + histogram
sns.distplot(test, hist=True, kde=True,
             bins=size, color='darkblue',
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 4})

# Add labels
plt.title('Density plot and Histogram of number of words of abstract')
plt.xlabel('Count of documents')
plt.ylabel('Number of words')
plt.show()


# histogram
plt.hist(test, color='blue', edgecolor='black',
         bins=size)
# Add labels
plt.title('Histogram of number of words of abstract')
plt.xlabel('Number of words')
plt.ylabel('Count of documents')
plt.show()


import plotly.figure_factory as ff
group_labels = ['distplot']  # name of the dataset
fig = ff.create_distplot([test], group_labels)
py.plot(fig, filename='schemas/distplot_word_count.html')


# ======================================================================================================================
# Find the maximum length of abstract in the whole dataset
# ======================================================================================================================

print("Maximum length of abstract in the whole dataset", max(data['abstract'].apply(len)))
