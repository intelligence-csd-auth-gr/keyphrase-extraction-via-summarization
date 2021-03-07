import json
import os
import sys
import gzip
import logging
from pke.base import LoadFile
from string import punctuation
from collections import defaultdict
from pandas import json_normalize
from pke import compute_document_frequency


# ======================================================================================================================
# Modified compute_document_frequency function to read from array/list/dataframe (instead of documents only)
# ======================================================================================================================

def compute_document_frequency_from_loaded_data(input_dir,
                                                output_file,
                                                language='en',
                                                normalization="stemming",
                                                stoplist=None,
                                                delimiter='\t',
                                                n=3,
                                                max_length=None,
                                                encoding=None):
    """Compute the n-gram document frequencies from a set of input documents. An
    extra row is added to the output file for specifying the number of
    documents from which the document frequencies were computed
    (--NB_DOC-- tab XXX). The output file is compressed using gzip.

    Args:
        input_dir (list): the input directory.
        output_file (str): the output file.
        language (str): language of the input documents (used for computing the
            n-stem or n-lemma forms), defaults to 'en' (english).
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        stoplist (list): the stop words for filtering n-grams, default to None.
        delimiter (str): the delimiter between n-grams and document frequencies,
            defaults to tabulation (\t).
        n (int): the size of the n-grams, defaults to 3.
        encoding (str): encoding of files in input_dir, default to None.
    """

    # document frequency container
    frequencies = defaultdict(int)

    # initialize number of documents
    nb_documents = 0

    # loop through the documents
    for input_file in input_dir:

        #logging.info('reading file {}'.format(input_file))

        # initialize load file object
        doc = LoadFile()

        # read the input file
        doc.load_document(input=input_file,
                          language=language,
                          normalization=normalization,
                          max_length=max_length,
                          encoding=encoding)

        # candidate selection
        doc.ngram_selection(n=n)

        # filter candidates containing punctuation marks
        doc.candidate_filtering(stoplist=stoplist)

        # loop through candidates
        for lexical_form in doc.candidates:
            frequencies[lexical_form] += 1

        nb_documents += 1

        if nb_documents % 1000 == 0:
            logging.info("{} docs, memory used: {} mb".format(nb_documents, sys.getsizeof(frequencies) / 1024 / 1024))

    # create directories from path if not exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # dump the df container
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:

        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(nb_documents)
        f.write(first_line + '\n')

        for ngram in frequencies:
            line = ngram + delimiter + str(frequencies[ngram])
            f.write(line + '\n')


# ======================================================================================================================
# Define stop word list
# ======================================================================================================================

# stoplist for filtering n-grams
stoplist = list(punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']


# ======================================================================================================================
# Compute document frequency for SemEval
# ======================================================================================================================

"""Compute Document Frequency (DF) counts from a collection of documents.

N-grams up to 3-grams are extracted and converted to their n-stems forms.
Those containing a token that occurs in a stoplist are filtered out.
Output file is in compressed (gzip) tab-separated-values format (tsv.gz).
"""

compute_document_frequency(input_dir='../data/benchmark_data/semeval_2010/train_test_combined/',
                           output_file='doc_freq/semeval_2010_doc_freq.tsv.gz',
                           extension='xml',           # input file extension
                           language='en',             # language of files
                           normalization="stemming",  # use porter stemmer
                           stoplist=stoplist)


# ======================================================================================================================
# Compute document frequency for SemEval
# ======================================================================================================================

file = '..\\data\\benchmark_data\\NUS.json'  # TEST data to evaluate the final model

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)

for index, abstract in enumerate(data['abstract']):
    # combine title + abstract + fulltext
    title_abstract = data['title'][index] + '. ' + abstract + data['fulltext'][index]
    # remove '\n'
    title_abstract = title_abstract.replace('\n', ' ')
    data['abstract'].iat[index] = title_abstract


compute_document_frequency_from_loaded_data(input_dir=data['abstract'],
                                            output_file='doc_freq/nus_doc_freq.tsv.gz',
                                            language='en',             # language of files
                                            normalization="stemming",  # use porter stemmer
                                            stoplist=stoplist)


# ======================================================================================================================
# Compute document frequency for SemEval
# ======================================================================================================================

file = '..\\data\\benchmark_data\\ACM.json'  # TEST data to evaluate the final model

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)


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

    title_abstract_mainBody = title + ' ' + abstract + ' ' + main_body
    # remove '\n'
    title_abstract_mainBody = title_abstract_mainBody.replace('\n', ' ')
    # print('title + abstract', title_abstract)

    data['fulltext'].iat[index] = title_abstract_mainBody


# rename column "fulltext" to "abstract" for uniformity between datasets
data.rename(columns={"fulltext": "abstract"}, inplace=True)
print(data)


compute_document_frequency_from_loaded_data(input_dir=data['abstract'],
                                            output_file='doc_freq/acm_doc_freq.tsv.gz',
                                            language='en',             # language of files
                                            normalization="stemming",  # use porter stemmer
                                            stoplist=stoplist)
