import ast  # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
import pke
import json
import string
import pandas as pd
import traditional_evaluation
from nltk.corpus import stopwords
from pandas import json_normalize
from nltk.stem.snowball import SnowballStemmer as Stemmer


# ======================================================================================================================
# Set file names in which pre-processed data will be saved
# ======================================================================================================================

# human written abstract (+ optionally full-text)

file_abstract = '..\\data\\benchmark_data\\ACM.json'  # TEST data to evaluate the final model
# summarized document
file_summaries = '..\\data\\benchmark_data\\summarization_experiment\\ACM_summarized.csv'  # TEST data to evaluate the final model


'''
# human written abstract (+ optionally full-text)
file_abstract = '..\\data\\benchmark_data\\NUS.json'  # TEST data to evaluate the final model
# summarized document
file_summaries = '..\\data\\benchmark_data\\summarization_experiment\\NUS_summarized.csv'  # TEST data to evaluate the final model
'''


'''
# human written abstract (+ optionally full-text)
file_abstract = '..\\data\\benchmark_data\\semeval_2010.json'
# summarized document
file_summaries = '..\\data\\benchmark_data\\summarization_experiment\\SemEval-2010_summarized.csv'  # TEST data to evaluate the final model
'''


# ======================================================================================================================
# Read data
# ======================================================================================================================

# load abstract
json_data = []
for line in open(file_abstract, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_abstract = json_normalize(json_data)
print(data_abstract)

# load summaries
data_summaries = pd.read_csv(file_summaries, encoding="utf8")
print(data_summaries)


# ======================================================================================================================
# Combine the title and abstract (+ remove '\n')
# ======================================================================================================================

def combine_text(data, file):
    if 'ACM' in file and 'json' in file:
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

            # use only title and abstract
            title_abstract_summary = title + ' ' + abstract

            # remove '\n'
            title_abstract_summary = title_abstract_summary.replace('\n', ' ')
            # print('title + abstract', title_abstract)

            data['fulltext'].iat[index] = title_abstract_summary

        # rename column "fulltext" to "abstract" for uniformity between datasets
        data.rename(columns={"fulltext": "abstract"}, inplace=True)
    else:
        for index, abstract in enumerate(data['abstract']):
            # use only title and abstract
            title_abstract_summary = data['title'][index] + '. ' + abstract
            # remove '\n'
            title_abstract_summary = title_abstract_summary.replace('\n', ' ')
            #    print('title_abstract_mainBody', title_abstract)

            data['abstract'].iat[index] = title_abstract_summary
        if 'keywords' in data.columns:
            # rename column "keywords" to "keyword" for uniformity between datasets
            data.rename(columns={"keywords": "keyword"}, inplace=True)

    print(data)

    return data


data_abstract = combine_text(data_abstract, file_abstract)
data_summaries = combine_text(data_summaries, file_summaries)


# ======================================================================================================================
# Format keyphrases and retrieve document text
# ======================================================================================================================

def extract_keyphrases(data):
    gold_keyphrases = []  # save the gold keyphrases of documents
    pred_keyphrases = []  # save the predicted keyphrases of documents
    for indx, abstract_document in enumerate(data['abstract']):
        # print('train_test_combined/' + key + '.xml')
        # print(keyphrases_dictionary[key])

        gold_keyphrases.append([[Stemmer('porter').stem(keyword) for keyword in keyphrase.split()] for keyphrase in data['keyword'][indx].split(';')])  # split gold keywords to separate them from one another

        # ======================================================================================================================
        # MultipartiteRank Extractor
        # ======================================================================================================================

        # 1. create a MultipartiteRank extractor.
        extractor = pke.unsupervised.MultipartiteRank()

        # 2. load the content of the document.
        extractor.load_document(input=abstract_document,
                                normalization="stemming")

        # 3. select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.74,
                                      method='average')

        # 5. get the 10-highest scored candidates as keyphrases
        pred_kps = extractor.get_n_best(n=10)

        pred_keyphrases.append([kp[0].split() for kp in pred_kps])  # keep only the predicted keyphrase and discard the frequency number

    print(pred_keyphrases)
    print(gold_keyphrases)

    return pred_keyphrases, gold_keyphrases


# ======================================================================================================================
# Get predictions
# ======================================================================================================================

pred_keyphrases_abstract, gold_keyphrases = extract_keyphrases(data_abstract)
pred_keyphrases_summaries, _ = extract_keyphrases(data_summaries)  # gold_keyphrases are the same as those of abstract (same document)


# ======================================================================================================================
# Combine ABSTRACT + SUMMARIES (document text + keyphrases)
# ======================================================================================================================
print("data_summaries['abstract']")
print(data_summaries['abstract'])
# needed for extraction f1-score - gold keyphrases should be checked if they exist with both abstract and summary
for doc_index, test_summar in enumerate(data_summaries['abstract']):
    data_summaries['abstract'].iat[doc_index] = data_abstract['abstract'][doc_index] + ' ' + test_summar
print(data_summaries['abstract'])

print('pred_keyphrases_abstract')
print(pred_keyphrases_abstract)
# combine the predicted keyphrase of the abstract and the summaries
for doc_indx, pred_keyphrase_abstr in enumerate(pred_keyphrases_abstract):
    pred_keyphrase_abstr.extend(pred_keyphrases_summaries[doc_indx])
    pred_keyphrases_abstract[doc_indx] = pred_keyphrase_abstr
print(pred_keyphrases_abstract)

# ======================================================================================================================
# Evaluation
# ======================================================================================================================

# traditional evaluation the model's performance
# ("x_filename": does not matter in this case -> used to combine paragraphs and sentences to original documents)
traditional_evaluation.evaluation(y_pred=pred_keyphrases_abstract, y_test=gold_keyphrases, x_test=data_summaries, x_filename='')
