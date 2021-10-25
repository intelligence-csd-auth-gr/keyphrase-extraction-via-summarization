
# Keyphrase Extraction via Extractive Summarization

This repository is a product of the work **"Keyphrase Extraction via Extractive Summarization"** published at the NAACL's Scholarly Document Processing (SDP) workshop. The paper can be accessed [here](https://aclanthology.org/2021.sdp-1.6/).

## Paper Abstract
Automatically extracting keyphrases from scholarly documents leads to a valuable concise representation that humans can understand and machines can process for tasks, such as information retrieval, article clustering and article classification. This paper is concerned with the parts of a scientific article that should be given as input to keyphrase extraction methods. Recent deep learning methods take titles and abstracts as input due to the increased computational complexity in processing long sequences, whereas traditional approaches can also work with full-texts. Titles and abstracts are dense in keyphrases, but often miss important aspects of the articles, while full-texts on the other hand are richer in keyphrases but much noisier. To address this trade-off, we propose the use of extractive summarization models on the full-texts of scholarly documents. Our empirical study on 3 article collections using 3 keyphrase extraction methods shows promising results.






## Libraries Installation

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```


## Data Download


- [KP20K](https://drive.google.com/file/d/1z1JGWMnQkkWw_4tjptgO-dxXD0OeTfuP/view) - provided by [Deep Keyphrase Generation](https://github.com/memray/OpenNMT-kpg-release)
- [NUS](https://github.com/boudinfl/ake-datasets/tree/master/datasets/NUS)
- [ACM](https://github.com/boudinfl/ake-datasets/tree/master/datasets/ACM)
- [SemEval 2010](https://github.com/snkim/AutomaticKeyphraseExtraction)

Place KP20K datasets under the folder:

```bash
\data\
```

Place NUS, ACM and SemEval 2010 datasets under the folder:

```bash
\data\benchmark_data\
```

Download "glove.6B.100d.txt" GloVe embeddings and place them in the project folder under the path:

```bash
\GloVe\glove.6B\
```

## Folders

| Folder | Description     |
| :-------- | :------- |
| **data**      | contains all the datasets and also the necessary scripts to generate the test datasets (folder are split by the experiment) |
| **data_statistics**      | run any script in this folder to get statistics from each dataset |
| **unsupervised_models**      | contains code for MultipartiteRank and TF-IDF models |






## Run Locally - Execute the following files in the order presented



### Convert format of ACM and SemEval 2010 datasets

**Prepare the SemEval 2010** dataset

```bash
data/benchmark_data/semeval_2010/combine_semeval_dataset.py
```

**Prepare the ACM** dataset

```bash
data/benchmark_data/acm_parser.py
```



### Generate summarizations

**Generate summarizations** for ACM, NUS and SemEval datasets

```bash
acm_nus_semeval_summarization.py
```



### Data pre-processing for training dataset (kp500k)

**Clean duplicate documents** between the train set and each of the test sets

```bash
data/benchmark_data/clean_duplicate_papers.py
```

Prepare the **KP20k** datasets (train: kp527k, validation: kp20k-v, test: kp20k)

```bash
preprocessing_full.py
```

Prepare the **KP20k split into sentences** datasets (train: kp527k, validation: kp20k-v, test: kp20k)

```bash
preprocessing_sentences.py
```

Change sequence size of string data without needing to pre-process data again

```bash
load_preprocessed_data.py
```



### Data pre-processing for test datsets - Run all scripts in the folders (ACM, NUS, SemEval 2010)

Prepare the test datasets for the **first three paragraphs of the full-text** experiments

```bash
data/benchmark_data/first_paragraphs_fulltext/
```

Prepare the test datasets for the **complete abstract** experiments

```bash
data/benchmark_data/full_abstract/
```

Prepare the test datasets for the **full-text split into paragraphs** experiments

```bash
data/benchmark_data/paragraph_fulltext/
```

Prepare the test datasets for the **abstract split into sentences** experiments

```bash
data/benchmark_data/sentence_abstract/
```

Prepare the test datasets for the **full-text split into sentences** experiments

```bash
data/benchmark_data/sentence_fulltext/
```

Prepare the test datasets for the **summarization of the full-text** experiments

```bash
data/benchmark_data/summarization_experiment/
```



### Train the model


**Train Bi-LSTM-CRF** model (uncomment the proper dataset file paths to select the desired test sets)

```bash
bi_lstm_crf.py
```



### Load a trained model

Load a **trained model**

```bash
load_pretrained_model.py
```

Load **trained models** for the experiments of the **combined predictions of Abstract & Summaries**

```bash
combined_summary_abstract_load_pretrained_model.py
```
