import time
import json
from datetime import timedelta
from pandas import json_normalize
from extractive import ExtractiveSummarizer


from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================
# Define summarizer (transformersum)
# ======================================================================================================================

# using extractive model "distilroberta-base-ext-sum"
model = ExtractiveSummarizer.load_from_checkpoint("models\\epoch=3.ckpt")


# ======================================================================================================================
# ACM dataset
# ======================================================================================================================

file = 'datasets\\ACM.json'  # TEST data to evaluate the final model

# ======================================================================================================================
# Read data
# ======================================================================================================================

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)


# ======================================================================================================================
# Summarize abstract and full-text (+ remove section identifiers and '\n')
# ======================================================================================================================

# extract title
def extract_title(fulltext):
    # extract the title
    start_title = fulltext.find("--T\n") + len("--T\n")  # skip the special characters '--T\n'
    end_title = fulltext.find("--A\n")
    title = fulltext[start_title:end_title]

    return title


# extract title
data['title'] = data['fulltext'].apply(extract_title)


# count the running time of the program
start_time = time.time()


# extract abstract and full-text and create a summary
for index, fulltext in enumerate(tqdm(data['fulltext'])):
    # extract the abstract
    start_abstract = fulltext.find("--A\n") + len("--A\n")  # skip the special characters '--A\n'
    end_abstract = fulltext.find("--B\n")
    abstract = fulltext[start_abstract:end_abstract]
    # print('abstract', abstract)

    # extract the fulltext
    start_fulltext = fulltext.find("--B\n") + len("--B\n")  # skip the special characters '--B\n'
    end_fulltext = fulltext.find("--R\n")  # do not include references
    main_body = fulltext[start_fulltext:end_fulltext]

    abstract_mainBody = abstract + ' ' + main_body

    # remove '\n'
    abstract_mainBody = abstract_mainBody.replace('\n', ' ')
    # print('title + abstract', title_abstract)

    # summarize abstract and full-text
    summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)

    data['fulltext'].iat[index] = summarize_fulltext


# rename column "fulltext" to "abstract" for uniformity between datasets
data.rename(columns={"fulltext": "abstract"}, inplace=True)

print(data)
print(data['abstract'][0])
print(data['abstract'][50])



total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- ACM %s running time ---" % total_time)


# ======================================================================================================================
# Save summarized ACM data to file
# ======================================================================================================================

summarized_file = 'datasets\\summarized_text\\ACM_summarized.csv'  # TEST data to evaluate the final model

data[['title', 'abstract', 'keyword']].to_csv(summarized_file, index=False)






# ======================================================================================================================
# NUS dataset
# ======================================================================================================================

# reading the initial JSON data using json.load()
file = 'datasets\\NUS.json'  # TEST data to evaluate the final model


# ======================================================================================================================
# Read data
# ======================================================================================================================

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)



# count the running time of the program
start_time = time.time()


# ======================================================================================================================
# Summarize abstract and full-text (+ remove '\n')
# ======================================================================================================================

# extract abstract and full-text and create a summary
for index, abstract in enumerate(tqdm(data['abstract'])):
    # combine abstract + main body
    abstract_mainBody = abstract + ' ' + data['fulltext'][index]

    # remove '\n'
    abstract_mainBody = abstract_mainBody.replace('\n', ' ')

    # summarize abstract and full-text
    summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)

    data['abstract'].iat[index] = summarize_fulltext

print(data)
print(data['abstract'][0])
print(data['abstract'][50])



total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- NUS %s running time ---" % total_time)


# ======================================================================================================================
# Save summarized NUS data to file
# ======================================================================================================================

summarized_file = 'datasets\\summarized_text\\NUS_summarized.csv'  # TEST data to evaluate the final model

data[['title', 'abstract', 'keywords']].to_csv(summarized_file, index=False)







# ======================================================================================================================
# SemEval-2010 dataset
# ======================================================================================================================

# reading the initial JSON data using json.load()
file = 'datasets\\semeval_2010.json'  # TEST data to evaluate the final model


# ======================================================================================================================
# Read data
# ======================================================================================================================

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)



# count the running time of the program
start_time = time.time()


# ======================================================================================================================
# Summarize abstract and full-text (+ remove '\n')
# ======================================================================================================================

# extract abstract and full-text and create a summary
for index, abstract in enumerate(tqdm(data['abstract'])):
    # combine abstract + main body
    abstract_mainBody = abstract + ' ' + data['fulltext'][index]

    # remove '\n'
    abstract_mainBody = abstract_mainBody.replace('\n', ' ')

    # summarize abstract and full-text
    summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)

    data['abstract'].iat[index] = summarize_fulltext

print(data)
print(data['abstract'][0])
print(data['abstract'][50])



total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- SemEval-2010 %s running time ---" % total_time)


# ======================================================================================================================
# Save summarized NUS data to file
# ======================================================================================================================

summarized_file = 'datasets\\summarized_text\\SemEval-2010_summarized.csv'  # TEST data to evaluate the final model

data[['title', 'abstract', 'keyword']].to_csv(summarized_file, index=False)
