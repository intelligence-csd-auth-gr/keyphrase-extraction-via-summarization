import json
import string
from pandas import json_normalize

from tqdm import tqdm
tqdm.pandas()


# reading the initial JSON data using json.load()
file_kp20k = '..\\kp20k_training.json'  # TRAIN data
file_kp20k_val = '..\\kp20k_validation.json'  # VALIDATION data to tune model parameters
file_kp20k_test = '..\\kp20k_testing.json'  # TEST data to evaluate the final model

file_nus = 'NUS.json'  # TEST data to evaluate the final model
file_acm = 'ACM.json'  # TEST data to evaluate the final model
file_semeval = 'semeval_2010.json'  # TEST data to evaluate the final model


# ======================================================================================================================
# Punctuation clearing
# ======================================================================================================================

punctuation = string.punctuation + '\t' + '\n'
table = str.maketrans(punctuation, ' '*len(punctuation))  # OR {key: None for key in string.punctuation}

print(punctuation, 'LEN:', len(punctuation))


def remove_punct(text):
    clean_text = text.translate(table)
    return clean_text


# ======================================================================================================================
# Load KP training
# ======================================================================================================================

json_data = []
for line in open(file_kp20k, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_kp20k = json_normalize(json_data)

# print(data_kp20k)

# remove punctuation
data_kp20k['clean_title'] = data_kp20k['title'].apply(remove_punct)
# remove redundant whitespaces
data_kp20k['clean_title'] = data_kp20k["clean_title"].str.replace('\s+', ' ', regex=True)


# ======================================================================================================================
# Load KP validation
# ======================================================================================================================

json_data = []
for line in open(file_kp20k_val, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_kp20k_val = json_normalize(json_data)

# print(data_kp20k)

# remove punctuation
data_kp20k_val['title'] = data_kp20k_val['title'].apply(remove_punct)
# remove redundant whitespaces
data_kp20k_val['title'] = data_kp20k_val["title"].str.replace('\s+', ' ', regex=True)


# ======================================================================================================================
# Load KP test
# ======================================================================================================================

json_data = []
for line in open(file_kp20k_test, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_kp20k_test = json_normalize(json_data)

# print(data_kp20k)

# remove punctuation
data_kp20k_test['title'] = data_kp20k_test['title'].apply(remove_punct)
# remove redundant whitespaces
data_kp20k_test['title'] = data_kp20k_test["title"].str.replace('\s+', ' ', regex=True)


# ======================================================================================================================
# Load NUS
# ======================================================================================================================

json_data = []
for line in open(file_nus, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_nus = json_normalize(json_data)

# print(data_nus)

# remove punctuation
data_nus['title'] = data_nus['title'].apply(remove_punct)
# remove redundant whitespaces
data_nus['title'] = data_nus["title"].str.replace('\s+', ' ', regex=True)


# ======================================================================================================================
# Load ACM
# ======================================================================================================================

json_data = []
for line in open(file_acm, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_acm = json_normalize(json_data)

# print(data_acm)


# extract titles for raw text
for index, fulltext in enumerate(data_acm['fulltext']):
    # extract the title
    start_title = fulltext.find("--T\n") + len("--T\n")  # skip the special characters '--T\n'
    end_title = fulltext.find("--A\n")
    title = fulltext[start_title:end_title]
    # print('title', title)
    title = title.translate(table)  # remove punctuation
    data_acm['fulltext'].iat[index] = title

# rename column "fulltext" to "abstract" for uniformity between datasets
data_acm.rename(columns={"fulltext": "title"}, inplace=True)

# remove redundant whitespaces
data_acm['title'] = data_acm["title"].str.replace('\s+', ' ', regex=True)


# ======================================================================================================================
# Load SemEval
# ======================================================================================================================

json_data = []
for line in open(file_semeval, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_semeval = json_normalize(json_data)

# print(data_nus)

# remove punctuation
data_semeval['title'] = data_semeval['title'].apply(remove_punct)
# remove redundant whitespaces
data_semeval['title'] = data_semeval["title"].str.replace('\s+', ' ', regex=True)


# ======================================================================================================================
# Lower case titles
# ======================================================================================================================

data_kp20k["clean_title"] = data_kp20k["clean_title"].str.lower()
data_nus["title"] = data_nus["title"].str.lower()
data_acm["title"] = data_acm["title"].str.lower()
data_semeval["title"] = data_semeval["title"].str.lower()
data_kp20k_val["title"] = data_kp20k_val["title"].str.lower()
data_kp20k_test["title"] = data_kp20k_test["title"].str.lower()

print(data_kp20k["title"])
print(data_kp20k["clean_title"])
print(data_nus["title"])
print(data_acm["title"])
print(data_semeval["title"])
print(data_kp20k_val["title"])
print(data_kp20k_test["title"])


# ======================================================================================================================
# Check for duplicate documents (based on title matching)
# ======================================================================================================================

# add new column to mark and drop duplicate papers
data_kp20k['duplicate'] = 0

count_dupl_docs_nus = 0
count_dupl_docs_acm = 0
count_dupl_docs_semeval = 0
count_dupl_docs_val = 0
count_dupl_docs_test = 0
for kp20k_index, kp20k_title in enumerate(tqdm(data_kp20k['clean_title'])):
    for nus_index, nus_title in enumerate(data_nus['title']):
        if kp20k_title == nus_title:
            #print('Round 2 - Duplicate NUS ', kp20k_index, ' found!')
            #print(kp20k_title, ' == ', nus_title)
            count_dupl_docs_nus += 1
            data_kp20k['duplicate'].iat[kp20k_index] = 1  # mark duplicate documents

    for acm_index, acm_title in enumerate(data_acm['title']):
        if kp20k_title == acm_title:
            #print('Round 2 - Duplicate ACM ', kp20k_index, ' found!')
            count_dupl_docs_acm += 1
            data_kp20k['duplicate'].iat[kp20k_index] = 1  # mark duplicate documents

    for semeval_index, semeval_title in enumerate(data_semeval['title']):
        if kp20k_title == semeval_title:
            #print('Round 2 - Duplicate ACM ', kp20k_index, ' found!')
            count_dupl_docs_semeval += 1
            data_kp20k['duplicate'].iat[kp20k_index] = 1  # mark duplicate documents

    for val_index, val_title in enumerate(data_kp20k_val["title"]):
        if kp20k_title == val_title:
            #print('Round 2 - Duplicate VALIDATION ', kp20k_index, ' found!')
            count_dupl_docs_val += 1
            data_kp20k['duplicate'].iat[kp20k_index] = 1  # mark duplicate documents

    for test_index, test_title in enumerate(data_kp20k_test['title']):
        if kp20k_title == test_title:
            #print('Round 2 - Duplicate TEST ', kp20k_index, ' found!')
            count_dupl_docs_test += 1
            data_kp20k['duplicate'].iat[kp20k_index] = 1  # mark duplicate documents


print('NUS COUNT: ', count_dupl_docs_nus, ' ACM COUNT: ', count_dupl_docs_acm, ' SemEval COUNT: ', count_dupl_docs_semeval,
      'VAL COUNT: ', count_dupl_docs_val, ' TEST COUNT: ', count_dupl_docs_test)

# NUS COUNT:  134
# ACM COUNT:  14
# VAL COUNT:  140
# TEST COUNT:  137

# drop duplicate paper columns of training dataset
data_kp20k = data_kp20k[data_kp20k['duplicate'] != 1]
data_kp20k.reset_index(drop=True, inplace=True)
print(data_kp20k['duplicate'])

# ======================================================================================================================
# Save clean training dataset
# ======================================================================================================================

print(data_kp20k)

data_kp20k.drop(['clean_title'], axis=1, inplace=True)
data_kp20k.drop(['duplicate'], axis=1, inplace=True)

print(data_kp20k)

# write data to json file
data_kp20k.to_json('kp20k_training.json', orient='records',  lines=True)

# Initial: 530809
# Final: 530390
