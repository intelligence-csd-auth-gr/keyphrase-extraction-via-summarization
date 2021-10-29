
# Keyphrase Extraction via Extractive Summarization

This repository is a product of the work **"Keyphrase Extraction via Extractive Summarization"** published at the NAACL's Scholarly Document Processing (SDP) workshop. The paper can be accessed [here](https://aclanthology.org/2021.sdp-1.6/).

## Paper Abstract
Automatically extracting keyphrases from scholarly documents leads to a valuable concise representation that humans can understand and machines can process for tasks, such as information retrieval, article clustering and article classification. This paper is concerned with the parts of a scientific article that should be given as input to keyphrase extraction methods. Recent deep learning methods take titles and abstracts as input due to the increased computational complexity in processing long sequences, whereas traditional approaches can also work with full-texts. Titles and abstracts are dense in keyphrases, but often miss important aspects of the articles, while full-texts on the other hand are richer in keyphrases but much noisier. To address this trade-off, we propose the use of extractive summarization models on the full-texts of scholarly documents. Our empirical study on 3 article collections using 3 keyphrase extraction methods shows promising results.









## Libraries Installation

Clone the project

```bash
  git clone https://github.com/intelligence-csd-auth-gr/keyphrase-extraction-via-summarization.git
```

Install libraries (Python 3.7)

```bash
  pip install -r requirements.txt
```


## Data Download

### Automated Data Set-up

**Download all datasets (KP20K, NUS, ACM & SemEval) used in the experiments [here](https://drive.google.com/file/d/19v_gSkI0Qo_BXyNFS9VOc3-0GuELf_Fl/view?usp=sharing) OR follow the instructions below to set them up manually.** To place the data in the working project folder, paste the downloaded ``data/`` folder into the root folder of the project. The downloaded folder cointains all necessary folder under the ``/data/preprocessed_data/`` folder, so skip the folder creation step (see section "Folders" below).

### Manual Data Set-up

- [KP20K](https://drive.google.com/file/d/1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR/view) (source: [initial repository](https://github.com/memray/seq2seq-keyphrase) of paper [Deep Keyphrase Generation](http://memray.me/uploads/acl17-keyphrase-generation.pdf))
- [NUS](https://drive.google.com/file/d/1z1JGWMnQkkWw_4tjptgO-dxXD0OeTfuP/view) (source: [updated repository](https://github.com/memray/seq2seq-keyphrase) of paper [Deep Keyphrase Generation](https://github.com/memray/OpenNMT-kpg-release))
- [ACM](https://github.com/boudinfl/ake-datasets/tree/master/datasets/ACM)
- [SemEval 2010](https://github.com/boudinfl/ake-datasets/tree/master/datasets/SemEval-2010)

Place the ``KP20K`` datasets (``kp20k_training.json``, ``kp20k_validation.json``, ``kp20k_testing.json``) under the folder:

```bash
/data/
```

For the ``NUS`` dataset:
- move the file ``data/json/nus/nus_test.json`` to ``data/benchmark_data/`` and rename it to ``NUS.json``

For the ``ACM`` dataset (create folders if not existing):
- place the contents of ``src/all_docs_abstacts_refined.zip`` inside the folder ``data/benchmark_data/test_dataset_processing/ACM/``,
- place the file ``references/test.author.stem.json`` in ``data/benchmark_data/test_dataset_processing/ACM/all_keys_in_json/``

For the ``SemEval 2010`` dataset (create folders if not existing):
- place the contents of both the``train/`` and ``test/`` folders into the project folder ``data/benchmark_data/test_dataset_processing/semeval_2010/train_test_combined/``,
- manually merge the files ``references/train.combined.stem.json`` and ``references/test.combined.stem.json`` into a file named ``train_test.combined.stem.json``, and, place it in ``data/benchmark_data/test_dataset_processing/semeval_2010/``



**Convert format** of ``ACM`` and ``SemEval 2010`` datasets

- **Prepare the SemEval 2010** dataset

```bash
python data/benchmark_data/test_dataset_processing/combine_semeval_dataset.py
```

- **Prepare the ACM** dataset

```bash
python data/benchmark_data/test_dataset_processing/acm_parser.py
```



### Download Pre-trained Word Embeddings

Download "glove.6B/glove.6B.100d.txt" [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings and place them in the root folder of the project under the path:

```bash
/GloVe/glove.6B/
```



## Folders

| Folder | Description     |
| :-------- | :------- |
| **data**      | contains all the datasets and also the necessary scripts to generate the test datasets (folders are split by experiment) |
| **data_statistics**      | run any script in this folder to get statistics from each dataset |
| **unsupervised_models**      | contains code for MultipartiteRank and TF-IDF models |


**Create the following folders under the folder ``/data/preprocessed_data/`` to store the processed data for all experiments:** (Skip if data were downloaded with the "Automated Data Set-up" method)

```bash
/data/preprocessed_data/first_paragraphs_fulltext/
```

```bash
/data/preprocessed_data/full_abstract/
```

```bash
/data/preprocessed_data/paragraph_fulltext/
```

```bash
/data/preprocessed_data/sentence_abstract/
```

```bash
/data/preprocessed_data/sentence_fulltext/
```

```bash
/data/preprocessed_data/summarization_experiment/
```





## Run Locally - Execute the following files in the order presented




### Generate summarizations

**Generate summarizations** for ``ACM``, ``NUS`` and ``SemEval`` datasets

```bash
python acm_nus_semeval_summarization.py
```

Move the generated files (``ACM_summarized.csv``,  ``NUS_summarized.csv`` and  ``SemEval-2010_summarized.csv``) that contain the summarizations into the folder  ``data/benchmark_data/summarization_experiment/``



### Data pre-processing for training dataset (kp500k)

**Clean duplicate documents** between the train set and each of the test sets

```bash
python data/benchmark_data/clean_duplicate_papers.py
```

Prepare the **KP20k** datasets (train: kp527k, validation: kp20k-v, test: kp20k)

```bash
python preprocessing_full.py
```

Prepare the **KP20k split into sentences** datasets (train: kp527k, validation: kp20k-v, test: kp20k)

```bash
python preprocessing_sentences.py
```

Change sequence size of string data without needing to pre-process data again

```bash
python load_preprocessed_data.py
```



### Data pre-processing for test datsets (ACM, NUS, SemEval) - Run all scripts in the folders

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
python bi_lstm_crf.py
```




### Load a trained model

Load a **trained model**

```bash
python load_pretrained_model.py
```

Load **trained models** for the experiments of the **combined predictions of Abstract & Summaries**

```bash
python combined_summary_abstract_load_pretrained_model.py
```



## Citation

Please cite the following paper if you are interested in using our code.

```bash
@inproceedings{kontoulis2021keyphrase,
  title={Keyphrase Extraction from Scientific Articles via Extractive Summarization},
  author={Kontoulis, Chrysovalantis Giorgos and Papagiannopoulou, Eirini and Tsoumakas, Grigorios},
  booktitle={Proceedings of the Second Workshop on Scholarly Document Processing},
  pages={49--55},
  year={2021}
}
```
