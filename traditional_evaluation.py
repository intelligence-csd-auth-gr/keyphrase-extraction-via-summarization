import ast  # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
import json
import numpy as np
import pandas as pd
from operator import itemgetter
from pandas import json_normalize
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer

from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)


def evaluation(y_pred=None, y_test=None, x_test=None, x_filename=None, y_filename=None, paragraph_assemble_docs=None):
    """
    Evaluate the performance
    :param y_pred: the predicted labels
    :param y_test: the test labels
    :param x_filename: the name of the GOLD document text file - NEED TO MATCH THE LOADED FILE WHEN MAKING PREDICTIONS (default evaluation dataset is KP20K)
    :param y_filename: the name of the GOLD keyphrase file - NEED TO MATCH THE LOADED FILE WHEN MAKING PREDICTIONS (default evaluation dataset is KP20K)
    :param paragraph_assemble_docs: (ONLY FOR UNSUPERVISED METHODS) the indices to re-assemble first 3 paragraphs
    :return: -
    """

    if y_test is None:  # evaluate the Bi-LSTM-CRF + unsupervised methods
        # ======================================================================================================================
        # Load all validation target data (y_test\labels) data on memory (needed for evaluation)
        # ======================================================================================================================

        # read preprocessed document text (x) and preprocessed keyphrases (y)
        x_test = pd.read_csv(x_filename, encoding="utf8")
        y_test = pd.read_csv(y_filename, encoding="utf8")

        # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
        x_test['abstract'] = x_test['abstract'].map(ast.literal_eval)
        if 'SENTENC' in x_filename or 'SENTEC' in x_filename or 'PARAGRAPH' in x_filename:
            assembl_docs = y_test['assemble_documents_index']
        y_test = y_test['keyword'].map(ast.literal_eval)

        print(y_test)


        # ======================================================================================================================
        # Convert y_test and y_pred from categorical (two columns, 1 for each label) to a single value label (1 column)
        # ======================================================================================================================

        def pred2label(all_abstract_preds):
            '''
            Converts prediction set and test/validation set from two columns (one for each label value)
            to just one column with the number of the corresponding label
            [ initial array: [1, 0] => final array: [0] ]   -   [ initial array: [0, 1] => final array: [1] ]
            :param all_abstract_preds: array with predictions or test/validation set [documents/abstracts, number of words]
            :return: flattened array that contains the prediction for each word [number of total words of all abstracts]
            '''
            preds = []
            for abstract_preds in all_abstract_preds:
                # the position of the max value is corresponding to the actual label value (0: Non-KP, 1: KP)
                doc_preds = [np.argmax(word_pred) for word_pred in abstract_preds]
                preds.append(doc_preds)
            return preds

        y_pred = pred2label(y_pred)  # convert y_pred from categorical (two columns, 1 for each label) to a single value label


        # ======================================================================================================================
        # Extract keyphrases from the predicted set
        # ======================================================================================================================

        pred_keyphrase_list = []  # save all predicted keyphrases
        for doc_index, doc_prediction in enumerate(y_pred):  # iterate through predictions for documents
            document_keyphrases = []  # save the keyphrases of a document
            consecutive_keywords = []  # save consecutive keywords that form a keyphrase
            for word_index, word_prediction in enumerate(doc_prediction):  # iterate through predictions for WORDS of documents
                if word_index >= len(x_test['abstract'][doc_index]):
                    break  # check if the abstract reached to an end (padding adds more dummy words non existing in real abstract)
                if word_index:  # check if this is the FIRST WORD in the abstract [to avoid negative index value]
                    if doc_prediction[word_index - 1]:  # check if the previous word is a keyword
                        if word_prediction:  # check if the current word is a keyword
                            consecutive_keywords.append(x_test['abstract'][doc_index][word_index])
                    else:
                        if len(consecutive_keywords):  # save keyword list if exists (not empty list)
                            document_keyphrases.append(consecutive_keywords)
                        consecutive_keywords = []  # re-initialize (empty) list
                        if word_prediction:  # check if the current word is a keyword
                            consecutive_keywords.append(x_test['abstract'][doc_index][word_index])
                else:  # save the FIRST WORD of the abstract if it is a keyword
                    if word_prediction:  # check if the current word is a keyword
                        consecutive_keywords.append(x_test['abstract'][doc_index][word_index])

            if len(consecutive_keywords):  # save the keywords that occur in the END of the abstract, if they exist (not empty list)
                document_keyphrases.append(consecutive_keywords)

            pred_keyphrase_list.append(document_keyphrases)
    else:  # evaluate the unsupervised methods that use .xml files
        # tokenize the text
        x_test['abstract'] = x_test['abstract'].apply(lambda row: row.split())
        print(x_test)

        # define pred_keyphrase_list - contains words
        pred_keyphrase_list = y_pred
        # define y_test if full-text in paragraphs/stentences
        if 'SENTENC' in x_filename or 'SENTEC' in x_filename or 'PARAGRAPH' in x_filename:
            assembl_docs = paragraph_assemble_docs


    # ======================================================================================================================
    # Calculate metrics
    # ======================================================================================================================

    def calculate_metrics(y_test_set, pred_keyphrase_list_set, eval_method):
        """
        Calculate and print metrics
        :param y_test_set: GOLD set
        :param pred_keyphrase_list_set: PREDICTION set
        :param eval_method: the name of the evaluation method (exact/partial match)
        :return: -
        """
        TP = 0  # True Positive
        FP = 0  # False Positive
        FN = 0  # False Negative
        for index_pred, doc_pred in enumerate(pred_keyphrase_list_set):
            for key_test in y_test_set[index_pred]:
                #if any(key_test not in keyp for keyp in doc_pred):
                if key_test not in doc_pred:  # FN: keyphrases that exist in GOLD but not in PREDICTED
                    FN += 1
            if len(doc_pred):  # continue if prediction list is NOT empty | if prediction list is empty -> skip checking
                for key_pred in doc_pred:
                    #if any(key_pred in keyp for keyp in y_test_set[index_pred]):
                    if key_pred in y_test_set[index_pred]:  # TP: keyphrases that exist both in PREDICTED and GOLD
                        TP += 1
                    else:  # FP: keyphrases that exist in PREDICTED but not in GOLD (if key_pred not in y_test_set)
                        FP += 1
        precision = 0
        recall = 0
        f1_score = 0
        # print(TP, FN, FP)
        # print('precision=', TP / (TP + FP), 'recall=', TP / (TP + FN))
        if not (TP == FP == 0):
            precision = TP / (TP + FP)
        if not (TP == FN == 0):
            recall = TP / (TP + FN)
        if not (precision == recall == 0):
            f1_score = 2 * (precision * recall) / (precision + recall)

        print('\n' + eval_method)
        print('Precision: %.4f' % precision)
        print('Recall: %.4f' % recall)
        print('F1-score: %.4f\n' % f1_score)


    # ======================================================================================================================
    # Calculate NEW metrics (semi-exact matching)
    # ======================================================================================================================

    def calculate_semi_exact_match_metrics(y_test_set, pred_keyphrase_list_set, eval_method):
        """
        Calculate and print metrics
        :param y_test_set: GOLD set
        :param pred_keyphrase_list_set: PREDICTION set
        :param eval_method: the name of the evaluation method (exact/partial match)
        :return: -
        """
        # each 0 and 1 represents a keyphrase and the 0 means that it exists in gold/pred set, while 0 means it does not
        pred_list = []  # contains 0, 1 for predicted keyphrases depending on if a predicted keyphrase matches with a gold one
        gold_list = []  # contains 0, 1 for gold keyphrases depending on if a gold keyphrase matches with a predicted one
        for index_pred, doc_pred in enumerate(pred_keyphrase_list_set):
            pred_kps = [0] * len(doc_pred)  # initialize the list with 0s and length equal to the total predicted keyphrases
            gold_kps = [0.0] * len(y_test_set[index_pred])  # initialize the list with 0s and length equal to the total gold keyphrases

            if doc_pred:  # if predicted keyphrase set is not empty (the case of empty predicted keyphrase is handled by the initialization of pred_kps and gold_kps)
                # find if the gold keyphrases exist in the predicted set, and if so mark which gold and predicted keyphrases have a match
                for gold_kp_index, gold_keyphr in enumerate(y_test_set[index_pred]):
                    gold_keyphrase_tokens = gold_keyphr.split()
                    avg_coverage_ratio_list = []
                    gold_coverage_ratio_list=[]
                    for pred_kp in doc_pred:
                        kw_coverage = 0  # gold keyword coverage of a predicted keyphrase
                        for keyword_gold in gold_keyphrase_tokens:
                            if keyword_gold in pred_kp:
                                kw_coverage += 1
                        # a gold keyword might exist multiple times in a pred keyphrase, but with this approach we assume that it does not as this happens rarely
                        if len(pred_kp.split()):
                            pred_coverage_ratio = kw_coverage / len(pred_kp.split())  # calculate the ratio of the covered predicted kps
                        else:
                            pred_coverage_ratio = 0
                        if len(gold_keyphrase_tokens):
                            gold_coverage_ratio = kw_coverage / len(gold_keyphrase_tokens)  # calculate the ratio of the covered gold kps
                        else:
                            gold_coverage_ratio = 0
                        avg_coverage_ratio_list.append((gold_coverage_ratio + pred_coverage_ratio) / 2)  # save the average of the keyphrase coverage and the coverage ratio
                        gold_coverage_ratio_list.append(gold_coverage_ratio)
                        
                    # find the max average coverage ratio and its position on the list
                    max_index, max_avg_coverage_ratio_list = max(enumerate(avg_coverage_ratio_list), key=itemgetter(1))
                    if max_avg_coverage_ratio_list > 0.5:
                        # set 1 or the average value of keyphrase coverage and ratio for possibly more accurate results
               #         gold_kps[gold_kp_index] = 1  # set 1 the gold kp that matched to a predicted one
                        gold_kps[gold_kp_index] = gold_coverage_ratio_list[max_index] #max_avg_coverage_ratio_list # gold_coverage_ratio_list[gold_kp_index]
                        pred_kps[max_index] = 1  # set 1 the predicted kp that was matched with a gold one

            # save the kp predicted/gold matches of each document
            pred_list.extend(pred_kps)
            gold_list.extend(gold_kps)

        FN = gold_list.count(0)  # False Negative: keyphrases that exist in GOLD but not in PREDICTED
   #     TP = gold_list.count(1)  # True Positive: keyphrases that exist both in PREDICTED and GOLD
        TP = sum(gold_list)
        FP = pred_list.count(0)  # False Positive: keyphrases that exist in PREDICTED but not in GOLD (if key_pred not in y_test_set)

        precision = 0
        recall = 0
        f1_score = 0
        if not (TP == FP == 0):
            precision = TP / (TP + FP)
        if not (TP == FN == 0):
            recall = TP / (TP + FN)
        if not (precision == recall == 0):
            f1_score = 2 * (precision * recall) / (precision + recall)


        print('\n' + eval_method)
        print('Precision: %.4f' % precision)
        print('Recall: %.4f' % recall)
        print('F1-score: %.4f\n' % f1_score)


    # ======================================================================================================================
    # Get the SETS of (unique) keyphrases for predicted and gold set
    # ======================================================================================================================

    # assemble the sentences of a document into a whole document again (only for the SENTEC and PARAGRAPH)
    print(x_filename)
    if 'SENTENC' in x_filename or 'SENTEC' in x_filename or 'PARAGRAPH' in x_filename:
        y_test_set = []  # set of original/all GOLD keyphrases for each document
        y_test_set_extraction = []  # keep only the GOLD keyphrases that exist in their corresponding document
        pred_keyphrase_list_set = []  # set of PREDICTED keyphrases for each document
        gold_same_document_keyphrases = []  # save the gold keyphrases that are from the same document (only for the SENTEC and PARAGRAPH)
        gold_extraction_same_document_keyphrases = []  # save the gold keyphrases that are from the same document - extraction (only for the SENTEC and PARAGRAPH)
        pred_same_document_keyphrases = []  # save the pred keyphrases that are from the same document (only for the SENTEC and PARAGRAPH)

        for doc_index, doc in enumerate(y_test):  # get the set of GOLD keyphrases for each document
            # y gold set
            gold_document_keyphrases = []  # save the keyphrases of a document as strings (each keyphrase -> string)
            # y gold set - extraction
            gold_document_keyphrases_extraction = []  # save the keyphrases of a document as strings (each keyphrase -> string)
            # y predicted
            pred_document_keyphrases = []  # save the keyphrases of a document as strings (each keyphrase -> string)

            abstract_as_string = ' '.join([Stemmer('porter').stem(word) for word in x_test['abstract'][doc_index]])

            for tokenized_keyphrase in doc:
                keyphrase = ' '.join(tokenized_keyphrase)  # STEMMING is already applied

                gold_document_keyphrases.append(keyphrase.strip())

                if keyphrase.strip() in abstract_as_string:  # keep only keyphrases that exist in the text - keyphrase EXTRACTION
                    gold_document_keyphrases_extraction.append(keyphrase.strip())


            for tokenized_keyphrase in pred_keyphrase_list[doc_index]:
                keyphrase = ''
                for word in tokenized_keyphrase:
                    keyphrase += Stemmer('porter').stem(word) + ' '  # apply STEMMING
                pred_document_keyphrases.append(keyphrase.strip())


            # check if the previous sentence is in the same document (has the same document id) as the current
            if doc_index == 0:
                gold_same_document_keyphrases.extend(gold_document_keyphrases)
                gold_extraction_same_document_keyphrases.extend(gold_document_keyphrases_extraction)
                pred_same_document_keyphrases.extend(pred_document_keyphrases)
            elif assembl_docs[doc_index] == assembl_docs[doc_index - 1]:
                gold_same_document_keyphrases.extend(gold_document_keyphrases)
                gold_extraction_same_document_keyphrases.extend(gold_document_keyphrases_extraction)
                pred_same_document_keyphrases.extend(pred_document_keyphrases)
            else:  # different documents
                # save keyphrases for the previous document
                y_test_set.append(set(gold_same_document_keyphrases))  # get each keyphrase just once
                y_test_set_extraction.append(set(gold_extraction_same_document_keyphrases))
                pred_keyphrase_list_set.append(set(pred_same_document_keyphrases))  # get each keyphrase just once

                # create the new document keyphrase set
                gold_same_document_keyphrases = gold_document_keyphrases
                gold_extraction_same_document_keyphrases = gold_document_keyphrases_extraction
                pred_same_document_keyphrases = pred_document_keyphrases

            # save the keyphrases for the last document
            if (doc_index + 2) > len(pred_keyphrase_list):  # (+2 because counting starts from 0 and we want the next element as well)
                # save keyphrases for the current document
                y_test_set.append(set(gold_same_document_keyphrases))  # get each keyphrase just once
                y_test_set_extraction.append(set(gold_extraction_same_document_keyphrases))
                pred_keyphrase_list_set.append(set(pred_same_document_keyphrases))  # get each keyphrase just once


        # count all keyphrases and keyphrases existing in text
        keyphrase_counter = 0
        extraction_keyphrase_counter = 0
        for doc_idx, y_test_extraction_doc in enumerate(y_test_set_extraction):
            extraction_keyphrase_counter += len(y_test_extraction_doc)
            keyphrase_counter += len(y_test_set[doc_idx])
        print('existing keyphrases', extraction_keyphrase_counter)
        print('all keyphrases', keyphrase_counter)

    else:  # for the full-text documents
        y_test_set = []  # set of original/all GOLD keyphrases for each document
        y_test_set_extraction = []  # keep only the GOLD keyphrases that exist in their corresponding document
        for doc_index, test_doc in enumerate(y_test):  # get the set of GOLD keyphrases for each document
            extraction_document_keyphrases = []  # save the keyphrases that exist in text (extraction) of a document as strings (each keyphrase -> string)
            document_keyphrases = []  # save all keyphrases of a document as strings (each keyphrase -> string)

            abstract_as_string = ' '.join([Stemmer('porter').stem(word) for word in x_test['abstract'][doc_index]])

            for tokenized_keyphrase in test_doc:
                keyphrase = ' '.join(tokenized_keyphrase)  # STEMMING is already applied

                document_keyphrases.append(keyphrase.strip())

                if keyphrase.strip() in abstract_as_string:  # keep only keyphrases that exist in the text - keyphrase EXTRACTION
                    extraction_document_keyphrases.append(keyphrase.strip())
                    
            y_test_set.append(set(document_keyphrases))  # get each keyphrase just once
            y_test_set_extraction.append(set(extraction_document_keyphrases))  # get each keyphrase just once


        # count all keyphrases and keyphrases existing in text
        keyphrase_counter = 0
        extraction_keyphrase_counter = 0
        for doc_idx, y_test_extraction_doc in enumerate(y_test_set_extraction):
            extraction_keyphrase_counter += len(y_test_extraction_doc)
            keyphrase_counter += len(y_test_set[doc_idx])
        print('existing keyphrases', extraction_keyphrase_counter)
        print('all keyphrases', keyphrase_counter)


        pred_keyphrase_list_set = []  # set of PREDICTED keyphrases for each document
        for doc in pred_keyphrase_list:  # get the set of PREDICTED keyphrases for each document
            document_keyphrases = []  # save the keyphrases of a document as strings (each keyphrase -> string)
            for tokenized_keyphrase in doc:
                keyphrase = ''
                for word in tokenized_keyphrase:
                    keyphrase += Stemmer('porter').stem(word) + ' '  # apply STEMMING
                document_keyphrases.append(keyphrase.strip())
            pred_keyphrase_list_set.append(set(document_keyphrases))  # get each keyphrase just once


    # ======================================================================================================================
    # Exact Match - Model Evaluation
    # ======================================================================================================================

    # Exact Match: the keyphrases must be given as whole strings

    # extraction - only GOLD KPs existing in text
    calculate_metrics(y_test_set_extraction, pred_keyphrase_list_set, 'Exact Match - Extraction')
    # all GOLD KPs
    calculate_metrics(y_test_set, pred_keyphrase_list_set, 'Exact Match')


    # ======================================================================================================================
    # NEW METHOD - Semi-Exact Match - Model Evaluation
    # ======================================================================================================================

    # extraction - only GOLD KPs existing in text
    calculate_semi_exact_match_metrics(y_test_set_extraction, pred_keyphrase_list_set, 'Semi-exact Match - Extraction')
    # all GOLD KPs
    calculate_semi_exact_match_metrics(y_test_set, pred_keyphrase_list_set, 'Semi-exact Match')


    # ======================================================================================================================
    # Partial Match - Model Evaluation
    # ======================================================================================================================

    # Partial Match: the keyphrases must be given as a set of words

    # Get the sets of all gold keyphrases
    y_test_set_partial = []
    for doc in y_test_set:  # get the set of GOLD keyphrases for each document
        document_keywords = []
        for keyphrase in doc:
            keyphrase = word_tokenize(keyphrase)
            for word in keyphrase:
                document_keywords.append(word)
        y_test_set_partial.append(set(document_keywords))

    # Get the sets of all gold keyphrases existing in text (extraction)
    y_test_set_partial_extraction = []
    for doc in y_test_set_extraction:  # get the set of GOLD keyphrases for each document
        document_keywords = []
        for keyphrase in doc:
            keyphrase = word_tokenize(keyphrase)
            for word in keyphrase:
                document_keywords.append(word)
        y_test_set_partial_extraction.append(set(document_keywords))

    # Get the sets of all predicted keyphrases
    pred_keyphrase_list_set_partial = []
    for doc in pred_keyphrase_list_set:  # get the set of PREDICTED keyphrases for each document
        document_keywords = []
        for keyphrase in doc:
            keyphrase = word_tokenize(keyphrase)
            for word in keyphrase:
                document_keywords.append(word)
        pred_keyphrase_list_set_partial.append(set(document_keywords))

    # extraction - only GOLD KPs existing in text
    calculate_metrics(y_test_set_partial_extraction, pred_keyphrase_list_set_partial, 'Partial Match - Extraction')
    # all GOLD KPs
    calculate_metrics(y_test_set_partial, pred_keyphrase_list_set_partial, 'Partial Match')
