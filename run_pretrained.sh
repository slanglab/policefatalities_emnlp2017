#!/bin/bash
TEST='data/sentments/test.json'
echo TEST DATA $TEST
#python code/models/logreg/extrfeats.py $TEST --ngrams --deps
python code/models/pretrained.py $TEST test_ng_dep.mtx code/models/model-pre-trained.pkl
cat code/models/preds/finalpreds.json | python code/eval/evaluation.py