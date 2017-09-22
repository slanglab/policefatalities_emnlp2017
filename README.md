policefatalities_emnlp2017/
=====================
Replication software, data, and supplementary materials for the paper: Keith et al., EMNLP-2017, "Identifying civilians killed by police with distantly supervised entity-event extraction."

Contact: Katherine Keith (kkeith@cs.umass.edu), Brendan O'Connor (brenocon@cs.umass.edu)

- data/
    - sentments/
        - train.json, test.json: sentence-level train and test files after preproc
            - docid = docid_fragid_sentid, i.e. the first number before the hyphen is the document id matching docs/ the second number is the fragment number within the document, and the third number is the sentence number within that fragment (see preproc/ for how these fragments and sentences were segmented)
    - gold/
        - fatalencs/
            - fe-raw.csv: raw Fatal Encounters (FE) file downloaded Feb. 27, 2017
            - fe-all.json: our FE post-processing (with hapnis normalized names)
        - guardian/
            - guard-raw.csv: raw guardian 2016 data downloaded Jan. 1, 2017
            - guard-all.json: our guardian post-processing (with hapnis normalized names)

- code/
    - eval/
        - evaluation.py: prints out AUPRC and best F1 for a given model
    - models/
        - preds/
            - .json files with predictions for the six models in the paper  
        - logreg/
            - Logistic Regression model code. See `README.md` in this directory for further instructions and notes
        - cnn/
            - CNN model code. See `README.md` in this directory for further instructions and notes
    - preproc/
        - scrape/
            - Code which downloads articles found via Google News and adds them to a Postgres database
        - dedupe/
            - Code which removes duplicate sentences from the dataset.
        - sentment/
            - hap/ : HAPNIS name normalization code
            - normnames.py : name normalization
            - getsentment.py : matches extracted sentences against gold data

- requirements.txt : pip installed packages in requirements format

- run.sh : runs the entire model pipleine (with data-pre-processed)

- run_pretrained.sh : runs the model pipeline with the pre-trained model for the given test data 

TRAIN/TEST SPLIT
=======
Model train/test split of documents:
- Training: Jan. - Aug. 2016
- Testing: Sept. - Dec. 2016

EVALUATION
==========
The evalutation script prints out the AUPRC and best F1 for the predictions of a given model.

Example usage:

`cat code/models/preds/m1.json | python code/eval/evaluation.py`

The evaluation code requries predictions in the following json format (see `code/models/preds/m1.json` for example) with one dictionary per line and dicitionary keys:
- "id" : document-sentence id
- "weight" : prediction on that mention given by the model
- "name" : name of the potential victim associated with that mention

RUNNING MODELS
======
To run the current model (logistic regression with EM training) with train/test data after pre-processing: 

`./run.sh`

To run the pre-trained model on test data 

`./run_pretrained.sh`


MODEL PIPELINE OUTLINE
==========

Here's an outline for the pipeline for soft (EM-based logistic regression):

1. Extract features

`python code/models/logreg/extrfeats.py data/sentments/train.json --ngrams --deps`

`python code/models/logreg/extrfeats.py data/sentments/test.json --ngrams --deps`

2. Run through logistic regression

`python code/models/logreg/runlr.py data/sentments/train.json data/sentments/test.json train_ng_dep.mtx test_ng_dep.mtx` 

3. Evaluate 

`cat code/models/preds/em50.json | python code/eval/evaluation.py`


MENTION-LEVEL DATA
==========
The preprocessed data used for this paper is `data/sentments/train.json` and `data/sentments/test.json`.

For both files, each line corresponds to a single mention with dictionary keys:
- "docid": document id  
- "name": HAPNIS normalized firstname, lastname pair of that is mapped to the 'TARGET' symbol
- "names_org": un-normalized names corresponding to 'name' that originally appeared in the text
- "sentnames": other names in the mention that will be mapped to 'PERSON' symbol
- "downloadtime": time the document was downloaded
- "sent_org": original mention text
- "sent_alter": mention text with names replaced by 'TARGET' and 'PERSON' symbols
- "plabel": 1 if 'name' matches a gold standard victim name in Fatal Encounters, 0 otherwise

CODE DEPENDENCIES
============
Feature extraction requires 
- Standford CoreNLP (https://stanfordnlp.github.io/CoreNLP/)
- Standford CoreNLP pywrapper (https://github.com/brendano/stanford_corenlp_pywrapper)
