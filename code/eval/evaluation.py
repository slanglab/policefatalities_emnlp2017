from __future__ import division
from collections import defaultdict
import sys,random
import ujson as json

'''
evaluation script for police fatality models

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

alldata = [json.loads(line) for line in open("../../data/gold/fatalencs/fe-all.json")]
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

byname=defaultdict(list)
for line in sys.stdin:
    d=json.loads(line)
    byname[d['name']].append(d)

print "num sentences", sum(len(xs) for xs in byname.itervalues())

entpred=[]
import numpy as np
for name in byname:
    scores = np.array([float(d['weight']) for d in byname[name]])
    scores = np.clip(scores, 0, 1-1e-16)
    prob = -np.sum(np.log1p(-scores))
    entpred.append( (name,prob) )

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

fs = 2*precs*recs/(precs+recs)
print "Best recall", recs[-1]
print "Best F1", np.max(fs), "for (p,r)=", precs[np.argmax(fs)],recs[np.argmax(fs)]
print "AUC",np.mean(aucs), np.std(aucs)
