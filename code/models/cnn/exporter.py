'''various utilities for sending CNN predictions to Katie and EM CNN'''

import json
import glob
import numpy as np
import cPickle
import socket, os
from log import *
setup("NA")

def eval_epoch(epoch_no, base, gpu):
    ''''make CNN results in a dictionary'''
    output = []
    # test_predictions/test_predictions_gpu3_b0_e55
    glob_loc = "/mnt/" + base + "_predictions/" + base + "_predictions_gpu" + str(gpu) +"_b*_e" + str(epoch_no)
    logger.info("eval " + glob_loc)
    print glob_loc
    for fn in glob.glob(glob_loc):
        with open(fn, "r") as inf:
            dt = cPickle.load(inf)
            tracker = dt["tracker"]
            predictions = dt["predictions"]
            for i in range(len(predictions)):
                metadata = tracker[i]
                prediction = predictions[i][1] # this index is definitely correct. is you change to 0 AUC goes to hell
                metadata["weight"] = str(prediction) # add in prediction to metadata

                # ids that end in "r" are flipped training data and we dont really care what it nn preds for these
                # make_new_preds assumes it is predicting Nmention things
                if metadata["id"] != "PLACEHOLDER":
                    output.append(metadata)
    return output


def make_new_preds(latest_epoch, gpu):
    '''make new preds for EM CNN'''
    print "[*] Make new preds..."
    orig_train = [o.replace("\n", "") for o in open("train.json")]
    nn_preds = eval_epoch(latest_epoch, "train", gpu) # get CNN predictions
    nn_preds_l = {}
    for n in nn_preds:
        nn_preds_l[n["id"] + "|||" + n["name"]] = float(n["weight"])
    preds = np.zeros(len(orig_train))
    logger.info("making new preds")
    for lno, train_ln in enumerate(orig_train):
        train_itm = json.loads(train_ln)
        preds[lno]  = nn_preds_l[train_itm["docid"] + "|||" + train_itm["name"]]

    preds = np.nan_to_num(preds) # NANs here?
    # Katie's pred clipping 
    preds[preds > 1-1e-5] = 1-1e-5
    preds[preds < 1e-5] = 1e-5

    return preds

def write_epoch(results, epo, gpu):
    ''''just write a prediction to a file'''
    with open("/mnt/eval/out{}-{}.json".format(epo, gpu), "w") as outf:
        for ln in results:
           outf.write(json.dumps(ln) + "\n")

if __name__ == "__main__":
    for epo in range(0,101,1):
        for gpu in [1,2,3]:
            try:
                results = eval_epoch(epo, "test", gpu)
                with (open("/mnt/test_processed/out{}-{}.json".format(epo,gpu), "w")) as outf:
                    for ln in results:
                        if ln["id"][-1] != "r":
                            outf.write(json.dumps(ln) + "\n")
                        else:
                            pass # print ln
            except EOFError:
                print "skip", epo
