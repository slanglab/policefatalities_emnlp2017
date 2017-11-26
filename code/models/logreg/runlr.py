from __future__ import division
import argparse, json
import numpy as np
import sys
from collections import defaultdict
import pickle
import em 
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

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('train', type=str, help='training .json file, both 0 and 1 examples')
    arg_parser.add_argument('test', type=str, help='testing .json file, both 0 and 1 examples')
    arg_parser.add_argument('X_train', type=str, nargs='?', help='training feature matrix, like train.mtx')
    arg_parser.add_argument('X_test', type=str, nargs='?', help='testing feature matrix, like test.mtx')
    arg_parser.add_argument('--emiters', type=int, help='number of iters for EM', default=50)
    args = arg_parser.parse_args()

    EM_ITERS = args.emiters 

    #---FULL DATASET-----
    Y_train, E_train, Epos_train, toprint_train = read_file(args.train)
    Y_test, E_test, Epos_test, toprint_test = read_file(args.test)

    #----READ IN FEATURE MATRICES 
    X_train = io.mmread(args.X_train)
    X_test = io.mmread(args.X_test)

    print "----------EM--------------"
    emModel = em.go_em(X_train, X_test, Y_test, E_train, Epos_train, toprint_test, Niter=EM_ITERS)

    #pickle that iteration of models 
    pkl = 'lr-soft-final.pkl'
    w = open(pkl, 'w')
    pickle.dump(emModel, w)
    w.close()
    #print "model saved to ", pkl

if __name__ == "__main__":
    main()
