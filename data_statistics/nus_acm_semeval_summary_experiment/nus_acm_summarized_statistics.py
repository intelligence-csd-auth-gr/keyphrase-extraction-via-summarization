import re
import json
import string
import pandas as pd
from pandas import json_normalize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer


from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)


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



# delete newline and tab characters
newLine_tabs = '\t' + '\n'
newLine_tabs_table = str.maketrans(newLine_tabs, ' ' * len(newLine_tabs))


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



punctuation = string.punctuation
table = str.maketrans(punctuation, ' '*len(punctuation))


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




def tokenize_lowercase(text):
    '''
    Toekenize, stem and convert to lower case the text of documents
    :param text: text of a specific document
    :return: formatted text
    '''
    words = word_tokenize(text)  # tokenize document text
    # get words of all keyphrases in a single list
    formatted_tok_text = [Stemmer('porter').stem(word_token.lower()) for word_token in words]  # DO NOT STEM TEXT WORDS TO TRAIN THE CLASSIFIER
    formatted_text = ' '.join(formatted_tok_text)
    return formatted_text


# ======================================================================================================================
# ACM summarized
# ======================================================================================================================

def acm_summarized_statistics():
    # reading the initial JSON data using json.load()
    file = '..\\data\\benchmark_data\\summarization_experiment\\ACM_summarized.csv'  # TEST data to evaluate the final model

    # ======================================================================================================================
    # Read data
    # ======================================================================================================================

    data = pd.read_csv(file, encoding="utf8")
    print(data)

    # ======================================================================================================================
    # Split keyphrases list of keyphrases from string that contains all the keyphrases
    # ======================================================================================================================

    for index, keywords in enumerate(data['keyword']):
        data['keyword'].iat[index] = keywords.split(';')  # split keywords to separate them from one another

    # ======================================================================================================================
    # Isolate the title, abstract and the main body (+ remove section identifiers and '\n')
    # ======================================================================================================================

    # tokenize key-phrases and keep them categorized by document
    for index, abstract in enumerate(data['abstract']):
        title_summary = data['title'][index] + ' ' + abstract  # combine title + abstract + main body
        # remove '\n'
        title_summary = title_summary.replace('\n', ' ')

        data['abstract'].iat[index] = title_summary

    # ======================================================================================================================
    # Remove Contractions (pre-processing)
    # ======================================================================================================================

    # substitute contractions with full words
    data['abstract'] = data['abstract'].apply(replace_contractions)
    data['keyword'] = data['keyword'].apply(lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])

    # ======================================================================================================================
    # Remove punctuation (with whitespace) + digits (from ABSTRACT) + clean empty strings
    # ======================================================================================================================

    # remove parenthesis, brackets and their contents
    data['abstract'] = data['abstract'].apply(remove_brackets_and_contents)

    # remove references of publications (in document text)
    data['abstract'] = data['abstract'].apply(remove_references)

    # remove punctuation
    data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
    data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)

    # Replace the pure digit terms with DIGIT_REPL
    data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
    print('convert digits - abstract finish')

    # remove rows with empty and one word abstracts/sentences
    data = data[data['abstract'].str.strip().astype(bool)]
    data.reset_index(drop=True, inplace=True)

    # remove empty keyphrases
    data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
    # remove rows with empty keyphrases
    data = data[data['keyword'].map(len) > 0]

    # ======================================================================================================================
    # Tokenize each sentence + remove digits (from KEYPHRASES)
    # ======================================================================================================================

    # tokenize text
    data['abstract'] = data['abstract'].apply(tokenize_lowercase)
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
                tokens = [Stemmer('porter').stem(keyword.lower()) for keyword in tokens]  # stem + lower case
                tokens = ' '.join(tokens)
                keyphrases_list.append(tokens)

        data['keyword'].iat[index] = keyphrases_list

    # ======================================================================================================================
    # Count logistics
    # ======================================================================================================================

    acm_keywords_in_summary = 0  # the count of keywords in abstract
    acm_total_keywords = 0  # the count of all keywords
    for index, keywords in enumerate(data['keyword']):
        acm_total_keywords += len(keywords)
        # print('total_keywords', len(test))
        # print('total_keywords', test)

        for keyword in keywords:
            # check if keyword exists on abstract
            if keyword in data['abstract'][index]:
                acm_keywords_in_summary += 1
                # print(keyword)
                # print(data['abstract'][index])

    print('ACM summarized: ', acm_keywords_in_summary)
    print('ACM summarized - total keyphrases: ', acm_total_keywords)

    print('ACM summarized - count of keywords in abstract: ', acm_keywords_in_summary / acm_total_keywords)

    return acm_keywords_in_summary


# ======================================================================================================================
# NUS summarized
# ======================================================================================================================

def nus_summarized_statistics():
    # ======================================================================================================================
    # Set batch size and file names in which pre-processed data will be saved
    # ======================================================================================================================

    # reading the initial JSON data using json.load()
    file = '..\\data\\benchmark_data\\summarization_experiment\\NUS_summarized.csv'  # TEST data to evaluate the final model

    # ======================================================================================================================
    # Read data
    # ======================================================================================================================

    data = pd.read_csv(file, encoding="utf8")
    print(data)

    # ======================================================================================================================
    # Split keyphrases list of keyphrases from string that contains all the keyphrases
    # ======================================================================================================================

    for index, keywords in enumerate(data['keywords']):
        data['keywords'].iat[index] = keywords.split(';')  # split keywords to separate them from one another

    # ======================================================================================================================
    # Combine the title, abstract and main body (+ remove '\n')
    # ======================================================================================================================

    # tokenize key-phrases and keep them categorized by document
    for index, abstract in enumerate(data['abstract']):
        title_summary = data['title'][index] + '. ' + abstract  # combine title + abstract + main body
        # remove '\n'
        title_summary = title_summary.replace('\n', ' ')

        data['abstract'].iat[index] = title_summary

    # ======================================================================================================================
    # Remove Contractions (pre-processing)
    # ======================================================================================================================

    # substitute contractions with full words
    data['abstract'] = data['abstract'].apply(replace_contractions)
    data['keywords'] = data['keywords'].apply(lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])

    # ======================================================================================================================
    # Remove punctuation (with whitespace) + digits (from ABSTRACT) + clean empty strings
    # ======================================================================================================================

    # remove parenthesis, brackets and their contents
    data['abstract'] = data['abstract'].apply(remove_brackets_and_contents)

    # remove references of publications (in document text)
    data['abstract'] = data['abstract'].apply(remove_references)

    # remove punctuation
    data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
    data['keywords'] = data['keywords'].apply(keyword_remove_punct_and_non_ascii)

    # Replace the pure digit terms with DIGIT_REPL
    data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
    # Replace the combination of characters and digits with WORD_DIGIT_REPL
    #data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('.*\d+', token) else 'WORD_DIGIT_REPL' for token in text.split()]))  # remove spaces
    print('convert digits - abstract finish')

    # remove rows with empty and one word abstracts/sentences
    data = data[data['abstract'].str.strip().astype(bool)]
    # reset index as explode results in multiple rows having the same index
    data.reset_index(drop=True, inplace=True)

    # remove empty keyphrases
    data['keywords'] = data['keywords'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
    # remove rows with empty keyphrases
    data = data[data['keywords'].map(len) > 0]

    # ======================================================================================================================
    # Tokenize each sentence + remove digits (from KEYPHRASES)
    # ======================================================================================================================

    # tokenize text
    data['abstract'] = data['abstract'].apply(tokenize_lowercase)
    print('tokenization - abstract finish')

    # stem, tokenize and lower case keyphrases and keep them categorized by document
    for index, list_of_keyphrases in enumerate(data['keywords']):
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
                tokens = [Stemmer('porter').stem(keyword.lower()) for keyword in tokens]  # stem + lower case
                tokens = ' '.join(tokens)
                keyphrases_list.append(tokens)

        data['keywords'].iat[index] = keyphrases_list

    # ======================================================================================================================
    # Count logistics
    # ======================================================================================================================

    nus_keywords_in_summary = 0  # the count of keywords in abstract
    nus_total_keywords = 0  # the count of all keywords
    for index, keywords in enumerate(data['keywords']):
        nus_total_keywords += len(keywords)

        for keyword in keywords:
            # check if keyword exists on abstract
            if keyword in data['abstract'][index]:
                nus_keywords_in_summary += 1

    print('NUS summary: ', nus_keywords_in_summary)
    print('NUS summary - total keyphrases: ', nus_total_keywords)

    print('NUS summary - count of keywords in abstract: ', nus_keywords_in_summary / nus_total_keywords)

    return nus_keywords_in_summary


# ======================================================================================================================
# ACM summarized
# ======================================================================================================================

def semeval_summarized_statistics():
    # reading the initial JSON data using json.load()
    file = '..\\data\\benchmark_data\\summarization_experiment\\SemEval-2010_summarized.csv'  # TEST data to evaluate the final model

    # ======================================================================================================================
    # Read data
    # ======================================================================================================================

    data = pd.read_csv(file, encoding="utf8")
    print(data)

    # ======================================================================================================================
    # Split keyphrases list of keyphrases from string that contains all the keyphrases
    # ======================================================================================================================

    for index, keywords in enumerate(data['keyword']):
        data['keyword'].iat[index] = keywords.split(';')  # split keywords to separate them from one another

    # ======================================================================================================================
    # Isolate the title, abstract and the main body (+ remove section identifiers and '\n')
    # ======================================================================================================================

    # tokenize key-phrases and keep them categorized by document
    for index, abstract in enumerate(data['abstract']):
        title_summary = data['title'][index] + ' ' + abstract  # combine title + abstract + main body
        # remove '\n'
        title_summary = title_summary.replace('\n', ' ')

        data['abstract'].iat[index] = title_summary

    # ======================================================================================================================
    # Remove Contractions (pre-processing)
    # ======================================================================================================================

    # substitute contractions with full words
    data['abstract'] = data['abstract'].apply(replace_contractions)
    data['keyword'] = data['keyword'].apply(lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])

    # ======================================================================================================================
    # Remove punctuation (with whitespace) + digits (from ABSTRACT) + clean empty strings
    # ======================================================================================================================

    # remove parenthesis, brackets and their contents
    data['abstract'] = data['abstract'].apply(remove_brackets_and_contents)

    # remove references of publications (in document text)
    data['abstract'] = data['abstract'].apply(remove_references)

    # remove punctuation
    data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
    data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)

    # Replace the pure digit terms with DIGIT_REPL
    data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  # remove spaces
    print('convert digits - abstract finish')

    # remove rows with empty and one word abstracts/sentences
    data = data[data['abstract'].str.strip().astype(bool)]
    data.reset_index(drop=True, inplace=True)

    # remove empty keyphrases
    data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
    # remove rows with empty keyphrases
    data = data[data['keyword'].map(len) > 0]

    # ======================================================================================================================
    # Tokenize each sentence + remove digits (from KEYPHRASES)
    # ======================================================================================================================

    # tokenize text
    data['abstract'] = data['abstract'].apply(tokenize_lowercase)
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
                tokens = [Stemmer('porter').stem(keyword.lower()) for keyword in tokens]  # stem + lower case
                tokens = ' '.join(tokens)
                keyphrases_list.append(tokens)

        data['keyword'].iat[index] = keyphrases_list

    # ======================================================================================================================
    # Count logistics
    # ======================================================================================================================

    semeval_keywords_in_summary = 0  # the count of keywords in abstract
    semeval_total_keywords = 0  # the count of all keywords
    for index, keywords in enumerate(data['keyword']):
        semeval_total_keywords += len(keywords)
        # print('total_keywords', len(test))
        # print('total_keywords', test)

        for keyword in keywords:
            # check if keyword exists on abstract
            if keyword in data['abstract'][index]:
                semeval_keywords_in_summary += 1
                # print(keyword)
                # print(data['abstract'][index])

    print('SemEval summarized: ', semeval_keywords_in_summary)
    print('SemEval summarized - total keyphrases: ', semeval_total_keywords)

    print('SemEval summarized - count of keywords in abstract: ', semeval_keywords_in_summary / semeval_total_keywords)

    return semeval_keywords_in_summary



