import numpy as np
import scipy
import scipy.sparse
from collections import defaultdict
import argparse, json, pickle
from sklearn.linear_model import LogisticRegression

def train_logreg(_X, _Q, C=0.1):
    """
    _X: size (Nmentions by Nfeat)
    _Q: vector length Nmentions. for each, P(z_i=1|x, y).
    model: for example a LogisticRegression object.
    """
    ## copy with Q weights
    Nmention = _X.shape[0]
    X =  scipy.sparse.vstack((_X, _X))
    Y = np.concatenate((np.ones(Nmention), np.zeros(Nmention)))
    weights = np.concatenate((_Q, 1-_Q))

    model = LogisticRegression(C=C, solver='lbfgs')
    model = model.fit(X, Y, weights)
    print model

    newpreds = model.predict_proba(X)
    assert newpreds.shape==(Nmention*2, 2)
    # Wanna return for the original mention examples, similar to _Q, prob of being true
    newpreds = newpreds[:Nmention,1]
    return {'model':model,'preds':newpreds}

def binary_entropy(qq):
    # vector of indep binary probs
    # sum(-qq*log(qq)-(1-qq)*log(1-qq))
    # but zeros cause madness
    ee = 0.0
    for q in qq:
        if q==0: continue
        ee += -q*np.log(q) - (1-q)*np.log(q)
    return ee

def disj_inference(preds, E, Epos, eid2rows):
    # gives us the Q values i.e. q(z) = p(z | x, y)

    # E: length Nmention. entity ID for each mention
    # Epos: length Nentity.  whether each entity is positive or not.
    # preds: the P(z_i=1|x_i) prior probs for each mention
    Nmention = len(preds); assert Nmention==len(E)
    assert len(eid2rows)==len(Epos)

    ret_marginals = np.zeros(Nmention)
    for (eid,row_indices) in eid2rows.items():
        if not Epos[eid]:
            # set Q=0 for these cases. use initialization from above
            continue
        ent_ment_preds = preds[row_indices]
        marginals = infer_disj_marginals(ent_ment_preds)
        ret_marginals[row_indices] = marginals

    return ret_marginals

def infer_disj_marginals(priors):
    disj_prob = 1-np.prod(1-priors)
    if disj_prob==0: return priors
    return priors / disj_prob

def go_em(X, X_test, Y_test, E, Epos, toprint_test, C=0.1, Niter=5, trainfn=train_logreg):
    # X: rows = sents/mentions, columns= features???
    # E: the entity ID for each row. integers from 0 to (Nentity-1) ex: E[0] = "John Doe", E[1] = "Michael Brown"
    # Epos: dict, keys= entities, values= True/False matches the gold data, ex: {"Alton Sterling": True}
    eid2rows = defaultdict(list) #keys = entities, values = list of sentence numbers, ex: {"John Doe": [0, 3, 6 ...]}
    Nmention = X.shape[0]
    for i in xrange(Nmention):
        eid2rows[E[i]].append(i)
    eid2rows = {eid: np.array(inds, dtype=int) for (eid,inds) in eid2rows.items()}
    all_values = np.concatenate(eid2rows.values())
    assert min(all_values)==0
    assert max(all_values)==Nmention-1
    assert len(Epos)==len(eid2rows)

    print "%s mentions, %s entities" % (Nmention, len(eid2rows))

    # Initialize to pseudolabel
    init_preds = np.zeros(Nmention)
    for eid in eid2rows:
        if Epos[eid]:
            init_preds[eid2rows[eid]] = 1.0

    #iter 0 (before EM)
    xx = trainfn(X, init_preds, C=C)
    cur_preds = xx['preds']
    #save the sent-level predictions (to be used in eval2)
    output_file = 'code/models/preds/em0.json'
    save_sent_level(X_test, toprint_test, output_file, xx['model'])

    #-------now set up for EM--------
    ## cur_preds: current P(z_i=1 | x) predictions.
    ## as opposed to P(z_i=1 | x,y) full E-step posteriors.
    cur_preds = init_preds
    cur_preds[cur_preds > 1-1e-16] = 1-1e-16
    cur_preds[cur_preds < 1e-16] = 1e-16     

    for itr in xrange(1, Niter+1):
        print ""
        print "=== EM iter",itr

        # E-step
        Q = disj_inference(cur_preds, E, Epos, eid2rows)

        assert Q.shape[0]==Nmention
        print "ELBO (after E step), {0}".format(elbo(Q, cur_preds))

        # M-step
        xx = trainfn(X, Q, C=C)
        assert xx['preds'].shape == cur_preds.shape
        print "ELBO (after M step), {0}".format(elbo(Q, xx['preds']))
        print 'ave pos pred, p(z_i = 1) : ', np.sum(xx['preds']) / len(xx['preds'])
        
        #UPDATE CURRENT PREDICITONS
        cur_preds = xx['preds']

        #save the sent-level predictions (to be used in eval2)
        output_file = 'code/models/preds/em{0}.json'.format(itr)
        print output_file
        save_sent_level(X_test, toprint_test, output_file, xx['model'])

    return xx['model']

def dotlog_reg(r, s):
    # similar to np.dot(r, np.log(s)) but will deal with cases where s=0
    assert len(r) == len(s)
    ent = 0 #either (neg) entropy or (neg) cross-entropy  
    for i in range(len(r)):
        if s[i] == 0.0: continue
        ent += r[i]*np.log(s[i])
    return ent

def elbo(q, preds):
    '''
    q: this is the marginal posterior of each mention label q(z_i=1)=p(z_i =1 | x, y)
    preds: this is the prediction given by the training function (LR or CNN)
        P(z_i | x)
    '''
    assert q.shape[0] == preds.shape[0]

    #all predictions should be non-zero, otherwise we're calculating the cross-entropy incorrectly
    assert np.all(preds != 0.0)

    elbo = 0 
    #----z_i =1, we don't need to change q
    #weighted log likelihood (or neg. cross entropy) when z_i=1
    #elbo += np.dot(q, np.log(preds))
    elbo += dotlog_reg(q, preds)
    
    #(neg) entropy when z_i = 1
    #elbo -= np.dot(q, np.log(q))
    elbo -= dotlog_reg(q, q)
    
    #----z_i = 0 we need to take 1 - q
    #weighted log likelihood (or neg. cross entropy) when z_i=0
    #elbo += np.dot(1- q, np.log(1 - preds))
    elbo += dotlog_reg(1-q, 1-preds)
    
    #(neg) entropy when z_i =0 
    #elbo -= np.dot(1 - q, np.log(1- q))
    elbo -= dotlog_reg(1-q, 1-q)

    return elbo

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
