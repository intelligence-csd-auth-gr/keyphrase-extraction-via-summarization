import ast  # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
import tables
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

from tqdm import tqdm

tqdm.pandas()

pd.set_option('display.max_columns', None)


def evaluation(y_pred, pad_length, y_test_filename='data\\preprocessed_data\\y_TEST_data_preprocessed'):
    """
    Evaluate the performance - sequence labeling evaluation
    :param y_pred: the predicted labels
    :param pad_length: the length of padded sequences
    :param y_test_filename: the name of the GOLD keyphrase file - NEED TO MATCH THE LOADED FILE WHEN MAKING PREDICTIONS (default evaluation dataset is KP20K)
    :return: -
    """

    # ======================================================================================================================
    # Load all validation target data (y_test\labels) data on memory (needed for evaluation)
    # ======================================================================================================================

    y_test = pd.read_csv(y_test_filename, encoding="utf8")
    # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
    y_test['y_test_keyword'] = y_test['y_test_keyword'].map(ast.literal_eval)
    #print(y_test)


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
            doc_preds = [np.argmax(word_pred) for word_pred in abstract_preds]
            '''
            for word_pred in abstract_preds:
                # the position of the max value is corresponding to the actual label value (0: Non-KP, 1: KP)
                doc_preds.append(1 if np.argmax(word_pred) else 0)
            '''
            preds.append(doc_preds)
        return preds

    # print('BEFORE y_pred', y_pred)
    y_pred = pred2label(y_pred)  # convert y_test from categorical (two columns, 1 for each label) to a single value label
    #print('AFTER y_pred', y_pred)


    # ======================================================================================================================
    # Evaluation of LM-LSTM-CRF paper
    # ======================================================================================================================

    total_labels = 0
    correct_labels = 0
    test = []
    pred = []
    for doc_index, y_test_doc in enumerate(y_test['y_test_keyword']):
        length = len(y_test_doc)  # length of the current test document (without padded values)

        # if the document is larger than the length of padded documents, keep only the pad length (500 or 70)
        if length > pad_length:
            length = pad_length

        # flatten test and pred to calculate metrics
        test.extend(y_test_doc[:length])   # truncate to 500 or 70 if document exceeds that number of words
        pred.extend(y_pred[doc_index][:length])  # OR y_pred[doc_index, :length]  # remove the padding values

    '''
        gold = y_test_doc[:length]  # truncate to 500 or 70 if document exceeds that number of words
        best_path = y_pred[doc_index][:length]  # OR y_pred[doc_index, :length]  # remove the padding values
    
        total_labels += length
        correct_labels += np.sum(np.equal(best_path, gold))

    precision = overlap_count / float(guess_count)
    recall = overlap_count / float(gold_count)

    if precision == 0.0 or recall == 0.0:
        print('total', 0.0, 0.0, 0.0)
    else:
        f = 2 * (precision * recall) / (precision + recall)
        print('precision', precision)
        print('recall', recall)
        print('f1-score', f)
    '''
    # CALCULATE FOR EACH LABEL THE EVALUATION SCORES?

    y_test = test
    y_pred = pred
    #print(y_test)
    #print(y_pred)


    # ======================================================================================================================
    # Calculate metrics
    # ======================================================================================================================

    # pos_label: the label that the score is reported for (KP - keyphrase label is selected as it is more important)
    print("recall for label KP: {:.2%}".format(recall_score(y_test, y_pred, pos_label=1)))
    print("precision for label KP: {:.2%}".format(precision_score(y_test, y_pred, pos_label=1)))
    print("F1-score for label KP: {:.2%}".format(f1_score(y_test, y_pred, pos_label=1)))
