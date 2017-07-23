'''
Evaluation script for police fatality models.
It accepts mention-level predictions as input, applies NoisyOR calculation of
entity-level predictions, and computes 
  - Area under Precision-Recall curve (AUPRC aka PRAUC)
  - Max F1

USAGE:  
cat **PREDS**.json | python evaluation.py 
    -prints out best recall, best F1 and AUC

cat **PREDS**.json | python evaluation.py --ent --sent
    -printout of top sentences per entity 

Here **PREDS**.json is a json format with one dictionary per line and dicitionary keys
    "id" :: doc-sent id 
    "weight" :: weight on that mention given by the model 
    "name" :: name associated with that mention 

For example,
cat ../preds/m1.json | python evaluation.py
'''

from __future__ import division
from collections import defaultdict
import sys,random,os
import json

if sys.stdin.isatty():
    print "Mention predictions should be on stdin.\n"
    print __doc__.strip()
    sys.exit(1)

here = os.path.dirname(__file__)
fe_all_filename = os.path.join(here, "../../data/gold/fatalencs/fe-all.json")
alldata = [json.loads(line) for line in open(fe_all_filename)]
testents = set(d['name'] for d in alldata if '2016-09-01' <= d['date'] <= '2016-12-31')
histents= set(d['name'] for d in alldata if d['date'] < '2016-09-01')
histents -= testents  ## person killed in past with same name as test-set person: do NOT count as historical.
print "num in test set %s, num historical %s" % (len(testents), len(histents))

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--ent', action='store_true')
parser.add_argument('--sent', action='store_true')
args=parser.parse_args()
if args.sent: args.ent=True

print "Reading mention-level predictions from standard input"
byname=defaultdict(list)
for line in sys.stdin:
    d=json.loads(line)
    byname[d['name']].append(d)

print "Number of mentions:", sum(len(xs) for xs in byname.itervalues())

def neg_log_not_noisyor(scores):
    """
    Assume 'scores' are P(z_i=1) probabilities.
    Calculate (log of complement of NoisyOR prob):
    log P(y_e=0) = sum_{i in M(e)} log(P(z_i=0))
    Use log1p for numerical stability since log(P(z=0))=log(1-p(z=1))
    Clip P(z_i=1) from being exactly 1.0 to ensure non-infinities in the
    log(P(y=0)) aggregation.
    """
    scores = np.clip(scores, 0, 1-1e-16)
    logprob_y0 = np.sum(np.log1p(-scores))
    return -logprob_y0

entpred=[]
import numpy as np
for name in byname:
    scores = np.array([float(d['weight']) for d in byname[name]])
    neg_lp_y0 = neg_log_not_noisyor(scores)
    entpred.append( (name,neg_lp_y0) )

# Calculate AUC several times for different tie-breakings to make sure that's
# not inducing variability.
aucs=[]
for itr in xrange(10):
    entpred.sort(key=lambda (e,p): (-p, random.random()))
    tp,fp,fn=0,0,len(testents)

    precs=[]
    recs=[]

    rank=0
    rank_incl_hist=0
    for e,p in entpred:
        rank_incl_hist += 1
        if e not in histents:
            rank += 1
            if e in testents:
                tp += 1
                fn -= 1
            else:
                fp += 1
            precs.append(tp/(tp+fp))
            recs.append(tp/(tp+fn))

        # Everything after this point is diagnostic output -- irrelevant to
        # auc/f1 evaluation.
        if itr>0: continue
        if not args.ent: continue

        if e not in histents:
            print "%s  rank=%s(%s) pred=%.6f nment=%4s tp=%s fp=%s fn=%s p=%.6f r=%.6f \t %s" % (
                    "POS" if (e in testents) else "NEG", 
                    rank, rank_incl_hist, p, len(byname[e]), 
                    tp, fp, fn,
                    tp*1.0/(tp+fp), tp*1.0/(tp+fn),
                    e.encode("utf-8"))
        elif e in histents:
            print "HIST rank=%s(%s) pred=%.6f nment=%4s \t %s" % (rank, rank_incl_hist, p, len(byname[e]), e.encode("utf8"))
        if args.sent:
            sents = byname[e]
            sents.sort(key=lambda d: -d['weight'])
            inds = range(min(3, len(sents)))
            for si in inds:
                d=sents[si]
                text = d.get('sent_org',"") or d.get('sent_alter',"")
                print "\n SENT  pred={p:.3g} {si}/{nsent} {ds_id} ||| {text}".format(p=d['weight'], si=si+1, nsent=len(sents), 
                        ds_id=d.get('docid',""), text=text.strip().encode("utf8"))
            print

    recs=np.array(recs)
    precs=np.array(precs)
    aucs.append(np.trapz(precs,recs))

# fs = 2*precs*recs/(precs+recs)
fs = [2*p*r/(p+r) if p+r>0 else 0 for (p,r) in zip(precs,recs)]
assert np.max(recs)==recs[-1]
print "Best recall", np.max(recs)
print "Best F1", np.max(fs), "for (p,r)=", precs[np.argmax(fs)],recs[np.argmax(fs)]
print "AUC %s (stdev across tiebreakings: %s)" % (np.mean(aucs), np.std(aucs))
