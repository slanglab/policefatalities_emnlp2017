"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import sys
gpuno = sys.argv[2]  # what GPU number to use.
import json
import time
import gzip
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
from log import setup
setup(gpuno)
from log import *
import theano
import theano.tensor as T
import gzip
import re
import glob
import warnings
import sys
import time
from exporter import make_new_preds
warnings.filterwarnings("ignore")



def dotlog_reg(r, s):
    # similar to np.dot(r, np.log(s)) but will deal with cases where s=0
    assert len(r) == len(s)
    ent = 0
    for i in range(len(r)):
        if s[i] == 0.0: continue
        if np.isnan(s[i]): continue
        if np.isnan(r[i]): continue
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

    entropy = dotlog_reg(q, q) + dotlog_reg(1-q, 1-q)
    e_neg_ll = dotlog_reg(q, preds) + dotlog_reg(1-q, 1-preds)

    return elbo, entropy, e_neg_ll


def read_file(file_name):
    #reads in .json train data
    Y = [] #the pseudolabel classes
    info = [] #info that will be printed for this evaluation
    sents = []
    E = [] #entity ID for each mention
    Epos = {} #whether each entity is positive or not.
    #with codecs.open(file_name, 'r', 'utf-8') as pf:
    with open(file_name, "r") as r:
        for line in r:
            docid = json.loads(line)["docid"].decode('utf-8')
            plabel = float(json.loads(line)["plabel"])
            assert type(plabel) == float
            assert plabel == 0.0 or plabel == 1.0
            name = json.loads(line)["name"].decode('utf-8')
            sent = json.loads(line)["sent_alter"].decode('utf-8')
            info.append({"id": docid, "name": name})
            #info.append({"id": docid, "name": name, "sent": sent, "plabel": plabel})
            Y.append(plabel)
            E.append(name)
            if plabel == 1.0: Epos[name] = True
            else: Epos[name] = False
            sents.append(sent)
    Y = np.array(Y)
    assert len(Y) == len(info) == len(sents) == len(E)
    print "READ FILE"
    print "NUM SENTS in {0}: {1}".format(file_name, len(sents))
    return sents, Y, info, E, Epos


def init_curr_preds(Nmention, E, Epos):
    eid2rows = defaultdict(list) #keys = entities, values = list of sentence numbers, ex: {"John Doe": [0, 3, 6 ...]}
    for i in xrange(Nmention):
        eid2rows[E[i]].append(i)
    eid2rows = {eid: np.array(inds, dtype=int) for (eid,inds) in eid2rows.items()}
    # assert min(eid2rows)==0 #KAK: don't think this assertion is what we want, keys of eid2rows are names
    all_values = np.concatenate(eid2rows.values())
    assert min(all_values)==0
    # assert max(eid2rows)==len(eid2rows)-1 #KAK: again, don't think this assertion is what we want, keys of eid2rows are names
    assert max(all_values)==Nmention-1
    assert len(Epos)==len(eid2rows)

    print "%s mentions, %s entities" % (Nmention, len(eid2rows))

    # Initialize to pseudolabel
    init_preds = np.zeros(Nmention)
    for eid in eid2rows:
        if Epos[eid]:
            init_preds[eid2rows[eid]] = 1.0
    init_preds[init_preds > 1-1e-16] = 1-1e-16
    init_preds[init_preds < 1e-16] = 1e-16
    return init_preds, eid2rows


execfile("conv_net_classes.py")

import argparse

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


def disj_inference(preds, E, Epos, eid2rows):
	# E: length Nmention. entity ID for each mention
	# Epos: length Nentity.  whether each entity is positive or not.
	# preds: the P(z_i=1|x_i) prior probs for each mention
	Nmention = len(preds); assert Nmention==len(E)
	assert len(eid2rows)==len(Epos)

	## not sure if this will be necessary. could be.
	#preds[preds > 1-1e-5] = 1-1e-5
	#preds[preds < 1e-5] = 1e-5
	assert np.all(preds >= 0) and np.all(preds <= 1)

	ret_marginals = np.zeros(Nmention)
	for (eid,row_indices) in eid2rows.items():
		if not Epos[eid]:
			# set Q=0 for these cases. use initialization from above
			continue
		ent_ment_preds = preds[row_indices]
		marginals = infer_disj_marginals(ent_ment_preds)
		ret_marginals[row_indices] = marginals


	# more numeric insanity
	ret_marginals[np.where(ret_marginals > 1.0)] = 1.0
	ret_marginals[np.where(ret_marginals < 0.0)] = 0.0

	return np.nan_to_num(ret_marginals) # more NAN fixes

def infer_disj_marginals(priors):
    return infer_disj_marginals2(priors)

def infer_disj_marginals2(priors):
    disj_prob = 1-np.prod(1-priors)
    if disj_prob==0: return priors
    return priors / disj_prob

def pad_weights(ts, w):
    '''ts: training set, w: weights'''
    padding = len(ts[:,-1]) - len(w)
    assert len(ts[:,-1]) == len(w) + padding
    weights2 = np.lib.pad(w, (0,padding), 'constant', constant_values=(0, 0))
    return weights2


def train_conv_net(fold_i,
                   datasets,
                   trackers,
                   U,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   pickle_w=False,
                   mode="vanilla",
                   cur_preds=None,
                   E=None,
                   Epos = None,
                   eid2rows=None):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """

    rng = np.random.RandomState()
    img_h = len(datasets[0][0])-1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters

    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    wt = T.ivector('wt')
    idxs = T.ivector()

    Words = theano.shared(value = U, name = "Words")

    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    train = [o for o in open("train.json")]

    #uncomment this back on for EM
    Q = disj_inference(cur_preds, E, Epos, eid2rows)
    weights = np.concatenate((Q, 1-Q))

    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y,wt)
    dropout_cost = classifier.dropout_negative_log_likelihood(y,wt)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    train_set = datasets[0]

    # debug code
    if mode == "debug":
        for exampleno, example in enumerate(train_set):
            if exampleno < len(train):
                plabel = json.loads(train[exampleno])["plabel"]
                label = example[len(example)-1]
                assert plabel == label
                opposite = 1 - plabel
                fliplabel = train_set[exampleno + len(train)][len(example) -1]
                assert opposite == fliplabel


        for tno, t in enumerate(train):
            pl = json.loads(t)
            assert  round(json.loads(t)["plabel"], 2) == round(weights[tno], 2)
        assert np.sum(weights) == len(train)

    for i in range(len(train)):
        train_set[i][len(train_set[i]) - 1] =1
        train_set[i + len(train)][len(train_set[i]) - 1] =0

    # check that weights match vanilla
    pos = [o for o in np.where(weights > .01)[0] if o < len(train)]
    neg = [o - len(train) for o in np.where(weights > .01)[0] if o > len(train)] #for u in np.where(weights > .01):
    for p in pos:
        #print p
        assert json.loads(train[p])["plabel"] == 1
    for n in neg:
        assert json.loads(train[n])["plabel"] == 0
    weights[np.where(weights > .99)] = 1
    weights[np.where(weights < .01)] = 0
    train_set_orig = np.where(weights > .99)
    #weights = np.ones(len(train_set))
    for lno,ln in enumerate(train_set):
        if lno < len(train_set)/2:
            endx = len(ln) -1
            mirror = train_set[lno + len(train)]
            assert np.array_equal(ln[0:len(ln)-1], mirror[0:len(ln)-1])
    extra_data_num = batch_size - train_set[0].shape[0] % batch_size
    extra_data = train_set[:extra_data_num]
    new_data=np.append(train_set,extra_data,axis=0)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches))

    test_set_x = datasets[1][0][:,:img_h]  #TODO THIS 0 index only gets first of these
    test_set_y = np.asarray(datasets[1][0][:,-1],"int32") #TODO THIS 0 index only gets first of these

    train_batches = {}
    for i in range(len(datasets[2])):
        train_batches[i] = datasets[2][i][:,:img_h]  #fill a bunch of batches


    train_set = new_data[:n_train_batches*batch_size,:]

    weights = pad_weights(train_set, weights)
    permuted_weights = weights


    train_set_x, train_set_y, train_set_wt = shared_dataset((train_set[:,:img_h],train_set[:,-1], permuted_weights))
    n_val_batches = n_batches - n_train_batches

    #compile theano functions to get train/val/test errors
    test_model = theano.function([idxs], classifier.errors(y),
             givens={
                 x: train_set_x[idxs],
                 y: train_set_y[idxs]},
                                 allow_input_downcast=True)
    train_model = theano.function([idxs, wt], cost, updates=grad_updates,
          givens={
              x: train_set_x[idxs],
              y: train_set_y[idxs]},
                                  allow_input_downcast = True)
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    p_z_given_x = classifier.predict_p(test_layer1_input)  # AH 1/18
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)
    pred_model_all = theano.function([x], p_z_given_x, allow_input_downcast = True) # AH 1/18
    # note this is exactly the same as train model but w/ no update=
    cost_model = theano.function([index, wt], cost, givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)
    # start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    test_perf = 0
    cost_epoch = 0
    print "n_train_batches=", n_train_batches
    train_losses = []
    for batch in range(0, len(train_set_orig), batch_size):
        train_losses.append(test_model(train_set_orig[0][batch:batch+batch_size]))
    train_perf = 1 - np.mean(train_losses)
    counter_t = 0

    start = time.time()
    print "go..."


    while (epoch < n_epochs):
        train_cost = 0
        counter_t = 0
        start_time = time.time()
        epoch = epoch + 1
        avg_costs = 0
        Q_draw = permuted_weights/np.sum(permuted_weights)
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            counter_t += 1
            if counter_t % 100 == 0:
                end = time.time()
                print counter_t, n_train_batches, end - start
                start = time.time()
            draws = np.random.choice(len(Q_draw), batch_size, p=Q_draw)
            cost_epoch = train_model(draws, np.ones(batch_size))#train_set_wt[minibatch_index*batch_size:(minibatch_index+1)*batch_size])
            avg_costs += cost_epoch
            set_zero(zero_vec)
        print "calc train losses"
        logger.info('AVG EPOCH_LOSS,{},{}'.format(epoch, avg_costs/n_train_batches))


        train_losses = []
        for batch in range(0, len(train_set_orig), batch_size):
            train_losses.append(test_model(train_set_orig[0][batch:batch+batch_size]))
        train_perf = 1 - np.mean(train_losses)

        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100.))

        test_predictions = pred_model_all(test_set_x)

        '''do test set predictions'''
        if True: # legacy conditional

            for i in range(len(datasets[1])):
                with open("/mnt/data/test_batches_{}.p".format(i), "r") as inf:
                    batch = cPickle.load(inf)
                test_predictions_i = pred_model_all(batch) # datasets[1][i][:,:img_h])
                test_tracker_i = trackers["tracker_test"][i]
                if i % 10 == 0:
                    print "predicted", i
                with open("/mnt/test_predictions/test_predictions_gpu{}_b{}_e{}".format(gpuno,i,epoch), "w") as outf:
                    cPickle.dump({"predictions":test_predictions_i, "tracker": test_tracker_i}, outf)

            print mode
            if mode == "em":

                for i in range(len(train_batches)):
                    train_predictions_i = pred_model_all(train_batches[i])
                    train_tracker_i = trackers["tracker_train_batches"][i]
                    if i % 10 == 0:
                        print "training prediction", i
                    with open("/mnt/train_predictions/train_predictions_gpu{}_b{}_e{}".format(gpuno,i,epoch), "w") as outf:
                        cPickle.dump({"predictions":train_predictions_i, "tracker": train_tracker_i}, outf)
                logger.info("about to calc elbo| {}".format(epoch))
                a,b,c= elbo(Q, make_new_preds(epoch, gpuno))
                logger.info("elbo| {}, {}, {}, {}".format(epoch,a,b,c))
            if mode == "em" and epoch !=0 and epoch % 2 ==0:
                '''do train set ''predictions'' for EM '''

                cur_preds = make_new_preds(epoch, gpuno)
                Q = disj_inference(cur_preds, E, Epos, eid2rows)
                with open("/mnt/Q/" + str(epoch) + "-" + str(gpuno), "w") as outf:
                    cPickle.dump(Q, outf)
                weights = pad_weights(train_set, np.concatenate((Q, 1-Q)))
                permuted_weights = weights#[p]

        test_loss = test_model_all(test_set_x,test_set_y)
        test_perf = 1- test_loss
        logger.info("PERF|{}, {}, {}\n".format(epoch, test_perf, train_perf))

        test_loss = test_model_all(test_set_x,test_set_y)

    return test_perf

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y, data_wt = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_wt = theano.shared(np.asarray(data_wt, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_wt, 'int32')

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def run():
    print "loading data..."
    with gzip.open('mr.p', 'rb') as f:
        x = cPickle.load(f)
    with gzip.open("datasets.p", "rb") as f:
        datasets = cPickle.load(f)
    with gzip.open("trackers.p", "rb") as f:
        trackers = cPickle.load(f)
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    mode = sys.argv[1]
    assert mode == "em" or mode == "vanilla" or mode == "debug"
    non_static=True
    execfile("conv_net_classes.py")
    U = W
    pickle_w = False
    results = []
    r = range(0,1)
    max_l = [kk.replace("\n", "") for kk in open("max_l.txt")].pop()
    max_l = int(max_l)
    i = 0
    d = .5

    sents_train, Y_train, info_train, E, Epos  = read_file("train.json")
    if mode == "EM":
        em_iter = [l for l in open('em.iter')].pop()
        em_iter = int(em_iter)
        init_preds, eid2rows = init_curr_preds(len(sents_train),E, Epos)
    else:
        em_iter = "NA"
        sents_train, Y_train, info_train, E_train, Epos_train  = read_file("train.json")
        init_preds, eid2rows = init_curr_preds(len(sents_train), E, Epos)
    s = 9
    perf = train_conv_net(i, datasets,trackers,
                          U,
                          lr_decay=0.95,
                          filter_hs=[3,4,5],
                          conv_non_linear="relu",
                          hidden_units=[100,2],
                          shuffle_batch=True,
                          n_epochs=100,
                          sqr_norm_lim=s,
                          non_static=non_static,
                          pickle_w=pickle_w,
                          batch_size=50,
                          dropout_rate=[d],
                          mode=mode,
                          cur_preds=init_preds,
                          E=E,
                          Epos=Epos,
                          eid2rows=eid2rows
                          )
    print "cv: " + str(i) + ", perf: " + str(perf), ", dropout_rate:", d
    results.append(perf)

if __name__=="__main__":
    run()
