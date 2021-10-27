import json
import pandas as pd
from pandas import json_normalize
import xml.etree.ElementTree as etree


# ======================================================================================================================
# Set file names in which pre-processed data will be saved
# ======================================================================================================================

keyphrases_file = 'semeval_2010/train_test.combined.stem.json'


# ======================================================================================================================
# Read data
# ======================================================================================================================

with open(keyphrases_file, 'r', encoding="utf8") as json_file:
    json_data = json.load(json_file)  # loads
print(json_data)

# convert json to dataframe
keyphrases_dictionary = json_normalize(json_data)

print(keyphrases_dictionary)


# ======================================================================================================================
# Format keyphrases and retrieve document text
# ======================================================================================================================

list_of_document_title = []  # save the title of documents
list_of_document_abstract = []  # save the abstract of documents
list_of_document_text = []  # save the body of documents
list_of_document_keyphrases = []  # save the keyphrases of documents
for key in keyphrases_dictionary:
    # print('train_test_combined/' + key + '.xml')
    # print(keyphrases_dictionary[key])

    keyphrase_string = ''
    # format the keyphrases as key1;key2;key3 (required for preprocessing)
    for list_of_keyphrases in keyphrases_dictionary[key]:
        for keyphrase in list_of_keyphrases:
            for nested_kp in keyphrase:  # read the nested keyphrases - different versions of the same keyphrases ( e.g. "number of sensor", "sensor number")
                keyphrase_string += nested_kp + ';'
        list_of_document_keyphrases.append(keyphrase_string[:-1])  # [:-1] -> remove the ';' in the end


    # read the documents' text
    parser = etree.XMLParser()  # re-initialize parser for each document
    path = 'semeval_2010/train_test_combined/' + key + '.xml'

    sentences = []  # keep all the information of a document
    tree = etree.parse(path, parser)
    for sentence in tree.iterfind('./document/sentences/sentence'):
        # get the character offsets
        starts = [int(u.text) for u in
                  sentence.iterfind("tokens/token/CharacterOffsetBegin")]
        ends = [int(u.text) for u in
                sentence.iterfind("tokens/token/CharacterOffsetEnd")]
        sentences.append({
            "words": [u.text for u in
                      sentence.iterfind("tokens/token/word")],
            "lemmas": [u.text for u in
                       sentence.iterfind("tokens/token/lemma")],
            "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
            "char_offsets": [(starts[k], ends[k]) for k in
                             range(len(starts))]
        })
        sentences[-1].update(sentence.attrib)

    '''
    PUBLICATION SECTIONS:

    title
    abstract
    introduction
    background
    method
    related work
    conclusions
    '''

    title = ''
    abstract = ''
    body = ''
    for indx, sent in enumerate(sentences):
        if sentences[indx]['section'] == 'title':  # for the title
            title += ' ' + ' '.join(sentences[indx]['words'])
        elif sentences[indx]['section'] == 'abstract':  # for the abstract
            abstract += ' ' + ' '.join(sentences[indx]['words'])
        else:  # for the main body (everything else)
            body += ' ' + ' '.join(sentences[indx]['words'])

    list_of_document_title.append(title)
    list_of_document_abstract.append(abstract)
    list_of_document_text.append(body)


print(list_of_document_title)
print(list_of_document_abstract)
print(list_of_document_keyphrases)
print(len(list_of_document_text))
print(len(list_of_document_title))
print(len(list_of_document_abstract))
print(len(list_of_document_keyphrases))


# ======================================================================================================================
# Write data to json file
# ======================================================================================================================

df = pd.DataFrame({'title': list_of_document_title, 'abstract': list_of_document_abstract, 'fulltext': list_of_document_text, 'keyword': list_of_document_keyphrases})
print(df)

# write data to json file
df.to_json('../semeval_2010.json', orient='records',  lines=True)
