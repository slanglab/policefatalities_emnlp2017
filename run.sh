#!/bin/bash
TRAIN='data/sentments/train.json'
TEST='data/sentments/test.json'
echo TRAINING $TRAIN
echo TESTING $TEST
python code/models/logreg/extrfeats.py $TRAIN --ngrams --deps
python code/models/logreg/extrfeats.py $TEST --ngrams --deps
python code/models/logreg/runlr.py $TRAIN $TEST train_ng_dep.mtx test_ng_dep.mtx
cat code/models/preds/em50.json | python code/eval/evaluation.py