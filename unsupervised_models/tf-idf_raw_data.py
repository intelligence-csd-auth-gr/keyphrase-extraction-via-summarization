import re
import ast  # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
import pke
import json
import string
import pandas as pd
import traditional_evaluation
from nltk.corpus import stopwords
from pandas import json_normalize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer


# ======================================================================================================================
# Set file names in which pre-processed data will be saved
# ======================================================================================================================

# human written abstract (+ optionally full-text)

file = '..\\data\\benchmark_data\\ACM.json'  # TEST data to evaluate the final model

'''
# summarized document
file = '..\\data\\benchmark_data\\summarization_experiment\\ACM_summarized.csv'  # TEST data to evaluate the final model
'''

input_file = 'doc_freq/acm_doc_freq.tsv.gz'




'''
# human written abstract (+ optionally full-text)
file = '..\\data\\benchmark_data\\NUS.json'  # TEST data to evaluate the final model
'''

'''
# summarized document
file = '..\\data\\benchmark_data\\summarization_experiment\\NUS_summarized.csv'  # TEST data to evaluate the final model
'''

'''
input_file = 'doc_freq/nus_doc_freq.tsv.gz'
'''




'''
# human written abstract (+ optionally full-text)
file = '..\\data\\benchmark_data\\semeval_2010.json'
'''

'''
# summarized document
file = '..\\data\\benchmark_data\\summarization_experiment\\SemEval-2010_summarized.csv'  # TEST data to evaluate the final model
'''

'''
input_file = 'doc_freq/semeval_2010_doc_freq.tsv.gz'
'''



# "0": use title and ABSTRACT,    "1": use title, abstract, and, FULLTEXT,    "2": FIRST 3 paragraphs (use json files)
use_fulltext = 2

# Max Length of the paragraphs (First 3 Paragraphs)
max_len = 400  # 400  # 220

RUN PLAIN ABSTRACT -> FOR "EXTRACTION KEYPHRASES" AND "ALL KEYPHRASES" NUMBERS


# ======================================================================================================================
# Read data
# ======================================================================================================================

if 'json' in file:
    json_data = []
    for line in open(file, 'r', encoding="utf8"):
        json_data.append(json.loads(line))

    # convert json to dataframe
    data = json_normalize(json_data)
else:
    data = pd.read_csv(file, encoding="utf8")
    #data['abstract'] = data['abstract'].map(ast.literal_eval)
    #data['keywords'] = data['keywords'].map(ast.literal_eval)
print(data)


# ======================================================================================================================
# Combine the title and abstract (+ remove '\n')
# ======================================================================================================================

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

        # extract the fulltext
        start_fulltext = fulltext.find("--B\n") + len("--B\n")  # skip the special characters '--B\n'
        end_fulltext = fulltext.find("--R\n")  # do not include references
        main_body = fulltext[start_fulltext:end_fulltext]

        if use_fulltext:  # use title, abstract, and, fulltext
            title_abstract_mainBody = title + ' ' + abstract + ' ' + main_body
        else:  # use only title and abstract
            title_abstract_mainBody = title + ' ' + abstract

        # remove '\n'
        title_abstract_mainBody = title_abstract_mainBody.replace('\n', ' ')
        # print('title + abstract', title_abstract)

        data['fulltext'].iat[index] = title_abstract_mainBody

    # rename column "fulltext" to "abstract" for uniformity between datasets
    data.rename(columns={"fulltext": "abstract"}, inplace=True)
else:
    for index, abstract in enumerate(data['abstract']):
        if 'summarized' in file:
            title_abstract = data['title'][index] + '. ' + abstract
        else:
            if use_fulltext:  # use title, abstract, and, fulltext
                title_abstract = data['title'][index] + '. ' + abstract + data['fulltext'][index]
            else:  # use only title and abstract
                title_abstract = data['title'][index] + '. ' + abstract
        # remove '\n'
        title_abstract = title_abstract.replace('\n', ' ')
        #    print('title_abstract_mainBody', title_abstract)

        data['abstract'].iat[index] = title_abstract
    if 'keywords' in data.columns:
        # rename column "keywords" to "keyword" for uniformity between datasets
        data.rename(columns={"keywords": "keyword"}, inplace=True)

print(data)

# ======================================================================================================================
# First 3 Paragraphs Experiment
# ======================================================================================================================

if use_fulltext == 2:
    file = 'PARAGRAPH'  # used in "traditional_evaluation" to re-combine the paragraphs to the original documents

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
        paragraph_count = 0  # count the paragraphs created (take only the first paragraphs of the text, as they contain the most keyphrases)
        list_of_paragraphs = []
        for idx, sentence in enumerate(doc):
            current_paragraph_length += len(word_tokenize(sentence))  # tokenize sentences to count their words

            if current_paragraph_length <= max_len:
                paragraph += sentence + ' '

                if (idx + 2) > len(doc):  # if this is the last sentence of the document (+2 because counting from 0 and we want the next element)
                    # save the completed paragraph
                    list_of_paragraphs.append(paragraph.strip())
            else:  # length exceeds 220
                if (idx + 2) > len(doc):  # if this is the last sentence of the document (+2 because counting starts from 0 and we want the next element as well)
                    # paragraph += sentence + ' '  # add the last sentence if though the length of this paragraph will exceed the boundary

                    # save the completed paragraph
                    list_of_paragraphs.append(paragraph.strip())

                    paragraph_count += 1  # increase saved paragraphs by 1

                    if paragraph_count >= 3:  # select the first 3 paragraphs
                        return list_of_paragraphs  # if the paragraph number is reached, do not insert the last sentence

                    paragraph = sentence  # save the remained sentence as a paragraph

                    # save the final-sentence paragraph
                    list_of_paragraphs.append(paragraph.strip())

                    if len(word_tokenize(paragraph.strip())) > max_len:  # find the paragraphs surpassing the boundary
                        print('paragraph/sentence is surpassing length boundary', len(word_tokenize(paragraph.strip())))
                else:
                    # save the completed paragraph
                    list_of_paragraphs.append(paragraph.strip())

                    paragraph_count += 1  # increase saved paragraphs by 1

                    if paragraph_count >= 3:  # select the first 3 paragraphs
                        return list_of_paragraphs

                    # reset the length count of the current paragraph to the length of the current first added sentence
                    current_paragraph_length = len(word_tokenize(sentence))  # tokenize sentences to count its words

                    # reset the current paragraph to a new paragraph
                    paragraph = sentence + ' '

                    if len(word_tokenize(paragraph.strip())) > max_len:  # find the paragraphs surpassing the boundary
                        print('paragraph/sentence is surpassing length boundary', len(word_tokenize(paragraph.strip())))

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
print(data)


# ======================================================================================================================
# Format keyphrases and retrieve document text
# ======================================================================================================================

gold_keyphrases = []  # save the gold keyphrases of documents
print(data)
print(gold_keyphrases)

list_of_document_title = []  # save the title of documents
list_of_document_abstract = []  # save the abstract of documents
list_of_document_text = []  # save the body of documents
pred_keyphrases = []  # save the predicted keyphrases of documents
for indx, abstract_document in enumerate(data['abstract']):
    # print('train_test_combined/' + key + '.xml')
    # print(keyphrases_dictionary[key])

    #if 'json' in file:
    gold_keyphrases.append([[Stemmer('porter').stem(keyword) for keyword in keyphrase.split()] for keyphrase in data['keyword'][indx].split(';')])  # split gold keywords to separate them from one another

# ======================================================================================================================
# TF-IDF Extractor
# ======================================================================================================================

    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')

    # 1. create a TfIdf extractor.
    extractor = pke.unsupervised.TfIdf()
    #print(' '.join(abstract_document))
    print(abstract_document)
    # 2. load the content of the document.
    extractor.load_document(input=abstract_document,  # ' '.join(abstract_document
                            language='en',
                            normalization="stemming")

    # 3. select {1-3}-grams not containing punctuation marks as candidates.
    extractor.candidate_selection(n=3, stoplist=stoplist)

    # 4. weight the candidates using a `tf` x `idf`
    df = pke.load_document_frequency_file(input_file=input_file)
    extractor.candidate_weighting(df=df)

    # 5. get the 10-highest scored candidates as keyphrases
    pred_kps = extractor.get_n_best(n=10)

    pred_keyphrases.append([kp[0].split() for kp in pred_kps])  # keep only the predicted keyphrase and discard the frequency number

print(pred_keyphrases)
print(gold_keyphrases)

# ======================================================================================================================
# Evaluation
# ======================================================================================================================

# traditional evaluation the model's performance
if use_fulltext == 2:
    traditional_evaluation.evaluation(y_pred=pred_keyphrases, y_test=gold_keyphrases, x_test=data, x_filename=file, paragraph_assemble_docs=data['assemble_documents_index'])
else:
    traditional_evaluation.evaluation(y_pred=pred_keyphrases, y_test=gold_keyphrases, x_test=data, x_filename=file)
