import ast  # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer

from tqdm import tqdm
tqdm.pandas()

pd.set_option('display.max_columns', None)


def evaluation(y_pred, x_filename='data\\preprocessed_data\\x_TEST_preprocessed_TEXT', y_filename='data\\preprocessed_data\\y_TEST_preprocessed_TEXT'):
    """
    Evaluate the performance
    :param x_filename: the name of the GOLD document text file - NEED TO MATCH THE LOADED FILE WHEN MAKING PREDICTIONS (default evaluation dataset is KP20K)
    :param y_filename: the name of the GOLD keyphrase file - NEED TO MATCH THE LOADED FILE WHEN MAKING PREDICTIONS (default evaluation dataset is KP20K)
    :param y_pred: the predicted labels
    :return:
    """
    # ======================================================================================================================
    # Load all validation target data (y_test\labels) data on memory (needed for evaluation)
    # ======================================================================================================================

    # read preprocessed document text (x) and preprocessed keyphrases (y)
    x_test = pd.read_csv(x_filename, encoding="utf8")
    y_test = pd.read_csv(y_filename, encoding="utf8")

    # translate string back to list of lists (when reading dataframe, lists of lists are read as strings)
    x_test['abstract'] = x_test['abstract'].map(ast.literal_eval)
    y_test['keyword'] = y_test['keyword'].map(ast.literal_eval)

    # print(x_test)
    # print(y_test)

    '''
    def load_y_test(y_test_filename, batch_size, number_of_batches):
        """
        Load y_test for validation
        :param y_test_filename: the file name that contains pre-processed data of y_test
        :param batch_size: the size of each batch
        :param number_of_batches: the total number of batches
        :return: return y_test (y_test_flat) for validation
        """
    
        y_test_batches = []  # save y_test for validation
        current_batch_number = 0  # the identifier used for each batch of data (ex. 0, 10000, 20000, 30000, etc.)
        while True:
            # Read X batches for testing from file (pre-processed)
            with tables.File(y_test_filename + '.hdf', 'r') as h5f:
                y_test_batches.append(h5f.get_node('/y_data' + str(current_batch_number)).read())  # get a specific chunk of data
    
            # calculate the identifier of each batch of data
            if current_batch_number < batch_size * number_of_batches:
                current_batch_number += batch_size
            else:
                y_test_flat = [y_label for y_batch in y_test_batches for y_label in y_batch]  # flatten the y_test (20000, 2881, 2)
                print('y_test SHAPE AFTER', np.array(y_test_flat, dtype=object).shape)
                return y_test_flat
    
    
    y_test = load_y_test(y_test_filename, batch_size, validation_test_steps - 1)  # load y_test in memory
    '''

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
    # print('AFTER y_pred', y_pred)


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
                        #                        print(x_test['abstract'][doc_index])
                        #                        print(x_test['abstract'][doc_index][word_index])
                        consecutive_keywords.append(x_test['abstract'][doc_index][word_index])
                else:
                    if len(consecutive_keywords):  # save keyword list if exists (not empty list)
                        document_keyphrases.append(consecutive_keywords)
                    consecutive_keywords = []  # re-initialize (empty) list
                    if word_prediction:  # check if the current word is a keyword
                        consecutive_keywords.append(x_test['abstract'][doc_index][word_index])
            else:  # save the FIRST WORD of the abstract if it is a keyword
                if word_prediction:  # check if the current word is a keyword
                    #               print('HEREEEE', doc_index, word_index)
                    #               print(x_test['abstract'][doc_index])
                    consecutive_keywords.append(x_test['abstract'][doc_index][word_index])

        if len(consecutive_keywords):  # save the keywords that occur in the END of the abstract, if they exist (not empty list)
            document_keyphrases.append(consecutive_keywords)

        pred_keyphrase_list.append(document_keyphrases)



    # FIND IF ANY KEYPHRASES EXIST ON THE PREDICTION SET
    here = [1 if any(doc) else 0 for doc in y_pred]
    print('\ny_pred', np.array(y_pred, dtype=object).shape)
    if any(here):
        print('THERE ARE KEYPHRASES')
    else:
        print('THERE ARE NOOOOOOT KEYPHRASES')


    # ======================================================================================================================
    # For the model trained on SENTENCES -> COMPILE ALL SENTENCES TOGETHER
    # ======================================================================================================================

#    print('pred_keyphrase_list', pred_keyphrase_list)
#    print('\npred_keyphrase_list', np.array(pred_keyphrase_list, dtype=object).shape)

    '''
    #df = (df.groupby(['NETWORK']).agg({'APPLICABLE_DAYS': lambda x_set: x_set.update(yoursequenceofvalues)})
    x_test['keyword'] = y_test['keyword']
    x_test['pred_keyp'] = pred_keyphrase_list
    print(x_test)
    x_test = (x_test.groupby(['keyword']).agg({'pred_keyp': lambda x_set: x_set}))
    #x_test = x_test.groupby('keyword')['pred_keyp'].apply(list)
    print(x_test)
    '''

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
        for index_pred, doc_pred in tqdm(enumerate(pred_keyphrase_list_set)):
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
    # Get the SETS of (unique) keyphrases for predicted and gold set
    # ======================================================================================================================
    keyphrase_counter=0
    y_test_set = []  # set of GOLD keyphrases for each document
    for doc_index, doc in enumerate(y_test['keyword']):  # get the set of GOLD keyphrases for each document
        document_keyphrases = []  # save the keyphrases of a document as strings (each keyphrase -> string)

        abstract_as_string = ' '.join([Stemmer('porter').stem(word) for word in x_test['abstract'][doc_index]])

        for tokenized_keyphrase in doc:
            keyphrase = ' '.join(tokenized_keyphrase)  # STEMMING is already applied
            '''
            keyphrase = ''
            #            print(tokenized_keyphrase)
            for word in tokenized_keyphrase:
                #                print(word)
                keyphrase += word + ' '  # STEMMING is already applied
            '''


            if keyphrase.strip() in abstract_as_string:  # keep only keyphrases that exist in the text - keyphrase EXTRACTION
                keyphrase_counter += 1
                document_keyphrases.append(keyphrase.strip())
            #            print(document_keyphrases)
        y_test_set.append(set(document_keyphrases))  # get each keyphrase just once
    #print('existing keyphrases', keyphrase_counter)

    pred_keyphrase_list_set = []  # set of PREDICTED keyphrases for each document
    for doc in pred_keyphrase_list:  # get the set of PREDICTED keyphrases for each document
        document_keyphrases = []  # save the keyphrases of a document as strings (each keyphrase -> string)
        for tokenized_keyphrase in doc:
            keyphrase = ''
            for word in tokenized_keyphrase:
                keyphrase += Stemmer('porter').stem(word) + ' '  # apply STEMMING
            document_keyphrases.append(keyphrase.strip())
        pred_keyphrase_list_set.append(set(document_keyphrases))  # get each keyphrase just once

    # print y_test and y_pred
    for i in range(len(pred_keyphrase_list_set)):
        print('pred', pred_keyphrase_list_set[i])
        print('test', y_test_set[i])


    # ======================================================================================================================
    # Exact Match - Model Evaluation
    # ======================================================================================================================

    # Exact Match: the keyphrases must be given as whole strings
    calculate_metrics(y_test_set, pred_keyphrase_list_set, 'Exact Match')  # calculate and print metrics

    '''
    # pos_label: the label that the score is reported for (KP - keyphrase label is selected as it is more important)
    print("F1-score for label KP: {:.2%}".format(f1_score(y_test, pred_keyphrase_list, pos_label='KP')))
    '''

    # ======================================================================================================================
    # Partial Match - Model Evaluation
    # ======================================================================================================================

    # Partial Match: the keyphrases must be given as a set of words

    # Get the sets of keyphrases for predicted and gold set
    y_test_set_partial = []
    for doc in y_test_set:  # get the set of GOLD keyphrases for each document
        document_keywords = []
        for keyphrase in doc:
            keyphrase = word_tokenize(keyphrase)
            for word in keyphrase:
                document_keywords.append(word)
        y_test_set_partial.append(set(document_keywords))

    pred_keyphrase_list_set_partial = []
    for doc in pred_keyphrase_list_set:  # get the set of PREDICTED keyphrases for each document
        document_keywords = []
        for keyphrase in doc:
            keyphrase = word_tokenize(keyphrase)
            for word in keyphrase:
                document_keywords.append(word)
        pred_keyphrase_list_set_partial.append(set(document_keywords))

    calculate_metrics(y_test_set_partial, pred_keyphrase_list_set_partial, 'Partial Match')  # calculate and print metrics
