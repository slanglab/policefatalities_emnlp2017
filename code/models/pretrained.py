from __future__ import division 
import argparse, json
import numpy as np
import sys
from collections import defaultdict
import pickle
from scipy import sparse, io

def read_file(file_name):
    #reads in .json train data
    Y = [] #the pseudolabel classes
    E = [] #entity ID for each mention
    Epos = {} #whether each entity is positive or not. 
    toprint = []
    with open(file_name, 'r') as r:
        for line in r:
            obj = json.loads(line)
            docid = obj["docid"]
            plabel = float(obj["plabel"])
            assert type(plabel) == float
            assert plabel == 0.0 or plabel == 1.0
            name = obj["name"]
            Y.append(plabel)
            E.append(name)
            if plabel == 1.0: Epos[name] = True
            elif plabel ==0.0: Epos[name]= False
            toprint.append({'id': docid, 'name': name})
    Y = np.array(Y)
    assert len(Y) == len(E) == len(toprint)
    print "READ {0} with {1} files".format(file_name, len(Y))
    return Y, E, Epos, toprint

def save_sent_level(X_test, toprint_test, output, model):
    probs = model.predict_proba(X_test)
    w = open(output, 'w')
    assert len(toprint_test) == len(probs)
    for i, yprob in enumerate(probs):
        obj = toprint_test[i]
        obj ['weight'] = yprob[1]
        json.dump(obj, w)
        w.write('\n')
    w.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('test', type=str, help='training .json file, both 0 and 1 examples')
    arg_parser.add_argument('X_test', type=str, nargs='?', help='training feature matrix, like train.mtx')
    arg_parser.add_argument('model', type=str, nargs='?', help='pre-trained model')
    args = arg_parser.parse_args()

    Y_test, E_test, Epos_test, toprint_test = read_file(args.test)
    X_test = io.mmread(args.X_test)

    model = pickle.load(open(args.model, 'r'))

    output_file = 'code/models/preds/finalpreds.json'
    save_sent_level(X_test, toprint_test, output_file, model)