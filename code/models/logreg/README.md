Overview 
======
Pipeline for logistic regression
1. Extract features (extrfeats.py)
2. Run logistic regression with EM (runlr.py)

extrfeats.py
====
Extracts features to be used in logistic regession models. 

Run feature extractor from the main directory file. From this directory run

```
cd ../../../

python code/models/logreg/extrfeats.py data/sentments/train.json --ngrams --deps

python code/models/logreg/extrfeats.py data/sentments/test.json --ngrams --deps
```

For ngram features only:

```
cd ../../../

python code/models/logreg/extrfeats.py data/sentments/train.json --ngrams 

python code/models/logreg/extrfeats.py data/sentments/test.json --ngrams 
``` 

For dependency features only 

```
cd ../../../

python code/models/logreg/extrfeats.py data/sentments/train.json --deps

python code/models/logreg/extrfeats.py data/sentments/test.json --deps
``` 

Breakdown of features extracted

(A) ngrams

A1) N-grams vanilla: length 1, 2, 3

A2) N-grams plus POS tags 

A3) N-grams plus lenth 1, 2, 3 directionality and position from target i.e. "kill_-2" (left) or "shot_3" (right)

A4) POS context: concatenated POS tags TARGET-2, TARGET-1, TARGET TARGET+1, TARGET+2

A5) N-grams full information: (word@-1, pos@-1, word@0, pos@0, word@+1, pos@+1)

(B) dependencies (CoreNLP)

B1) length 3 dependencies that includes TARGET: words, POS, dep labels

B2) length 3 dependencies that includes TARGET: just words and dep labels

B3) length 3 dependencies that includes TARGET: just words and pos tags

B4) All dependencies length 2

runlr.py
=====

Runs code for hard-LR and soft-LR. 
Hard-LR is just iteration 1 of EM.

USEAGE: 

`mkdir preds/`

`python runlr.py train.json test.json train.mtx test.mtx` 

Output will be `preds/em0.json, preds/em1.json, ... preds/em50.json`

