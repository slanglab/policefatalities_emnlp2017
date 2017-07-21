'''
- this is a minor modification of Yoon Kims's process_data.py

- some of yoon kim's preprcessing steps from conv_net_sentence.py have
  also been moved in this this file, such as get_idx_from_sent
'''
import numpy as np
import gzip
import cPickle
from collections import defaultdict
import sys, re, glob
import ujson as json
import pandas as pd
import ipdb, gzip

DEBUG = False
FULL = 9
SPLIT_MONTH = FULL

'''
this code is for ensuring that the 'revs' from Yoon Kims preproc match the order
of train.json which was helpful for debugging
'''
Q_ids = []
q_id_to_loc = {}
with open("train.json", "r") as inf:
    for lno, ln in enumerate(inf):
        dt = json.loads(ln.replace("\n", ""))
        Q_ids.append(dt["name"] + "|||" + dt["docid"])
        q_id_to_loc[lno] = dt["name"] + "|||" + dt["docid"]

Q_ids = set(Q_ids)

def put_in_q_order(revs_, not_q):
    out = []
    lookup = {}
    for r in revs_:
        key_ = r["name"] + "|||" + r["id"]
        lookup[key_] = r
    for lno in range(len(revs_)):
        if lno % 100 == 0:
            print lno, len(revs_)
        next_up = q_id_to_loc[lno]
        if not_q:
            add_this = lookup[next_up + "r"]
        else:
            add_this = lookup[next_up]
        out.append(add_this)
    assert len(out) == len(revs_)
    return out

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)

    return x

'''
AH: I think this is legacy code
def get_ids_to_weights():
    id_to_weight = {}

    for fn in glob.glob("train_predictions/*"):
        with open(fn, "r") as inf:
            preds = cPickle.load(inf)
            tracker = preds["tracker"]
            predictions = preds["predictions"]
            for tno, t in enumerate(tracker):
                pred = predictions[tno]
                id_to_weight[t["id"]] = str(pred[0])
    return id_to_weight
'''

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    tracker: keeps tabs on ids -AH:1/20
    """
    train, test = [], []
    tracker_train, tracker_test = [], []
    test_batch = [] # you need to run the test in batches
    test_batch_tracker = []

    train_batches = []
    train_batches_tracker = []
    train_batch = []
    train_batch_tracker = []
    MAXTEST = 1000
    print "[*] make idx data cv"
    for revno, rev in enumerate(revs):
        if revno % 50000 == 0:
            print revno
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        id_ = rev["id"]
        name = rev["name"]
        olno = rev["orig_line_no"]
        if rev["split"]=="test":
            if len(test_batch) < MAXTEST:
                test_batch.append(sent)
                test_batch_tracker.append({"id": id_, "name": name, "orig_line_no": olno})
            else:
                test.append(test_batch)
                tracker_test.append(test_batch_tracker)
                test_batch = []
                test_batch_tracker = []
                test_batch.append(sent)
                test_batch_tracker.append({"id": id_, "name": name, "orig_line_no": olno})
        else:
            train.append(sent)
            tracker_train.append({"id":id_, "name": name, "orig_line_no": olno})
            if len(train_batch) < MAXTEST:
                train_batch.append(sent)
                train_batch_tracker.append({"id": id_, "name": name, "orig_line_no":olno})
            else:
                train_batches.append(train_batch)
                train_batches_tracker.append(train_batch_tracker)
                train_batch = [sent]
                train_batch_tracker = [{"id": id_, "name": name, "orig_line_no": olno}]

    # catch any stragglers in incomplete batches
    if len(test_batch) > 0:
        while len(test_batch) < MAXTEST:
            test_batch.append(sent)
            test_batch_tracker.append({"id": "PLACEHOLDER", "name": "NA"})
        test.append(test_batch) # add any remainders
        tracker_test.append(test_batch_tracker)

    if len(train_batch) > 0:
        while len(train_batch) < MAXTEST:
            train_batch.append(sent)
            train_batch_tracker.append({"id": "PLACEHOLDER", "name": "NA"})
        train_batches.append(train_batch) # add any remainders
        train_batches_tracker.append(train_batch_tracker)


    train = np.array(train, dtype="int")
    test = [np.array(t, dtype="int") for t in test]
    train_batches = [np.array(t, dtype="int") for t in train_batches]

    # train_batches is used to make predictions on the E_step
    return [train, test, train_batches], {"tracker_train": tracker_train, "tracker_test":tracker_test, "tracker_train_batches": train_batches_tracker} # ah trackers added


def build_data_cv(data_folder, cv=10, clean_string=False):
    """
    Loads data. cv=10 is from yoon kim's original code. We don't use it.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    Nmentions = 0
    with open(pos_file, "rb") as f:
        for lno, line in enumerate(f):
            dt = json.loads(line.replace("\n", ""))
            id_, line, name, scrape_month = dt["doc_id"], dt["sent_alter"], dt["name"], int(dt["scrape_month"])
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())

            if len(orig_rev.split()) < 200:
                for word in words:
                    vocab[word] += 1
                split = "train" if scrape_month < SPLIT_MONTH else "test"
                if split == "train":
                    Nmentions += 1
                datum  = {"id":id_,
                          "y":1,
                          "orig_line_no": dt["lno"],
                          "name": name,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": split}
                revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            dt = json.loads(line.replace("\n", ""))
            id_, line, name, scrape_month = dt["doc_id"], dt["sent_alter"], dt["name"], int(dt["scrape_month"])
            split = "train" if scrape_month < SPLIT_MONTH else "test"
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            words = set(orig_rev.split())
            if len(orig_rev.split()) < 200:
                if split == "train":
                    Nmentions += 1
                for word in words:
                    vocab[word] += 1
                datum  = {"id": id_,
                          "y":0,
                          "orig_line_no": dt["lno"],
                          "name": name,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": split}
                revs.append(datum)
    with open("Nmentions.txt", "w") as outf:
        outf.write(str(Nmentions))

    revs_train = [r for r in revs if r["split"] == "train"]
    revs_test = [r for r in revs if r["split"] == "test"]

    revs_q = [r for r in revs_train if r["name"] + "|||" + r["id"] in Q_ids]
    revs_not_q = [r for r in revs_train if r["name"] + "|||" + r["id"] not in Q_ids]
    if DEBUG == True:
        assert len(revs_q) + len(revs_not_q) == len(revs_train)
        assert len(revs_train) + len(revs_test) == len(revs)

    revs_train = put_in_q_order(revs_q, False) + put_in_q_order(revs_not_q, True)
    with open("train.json") as inf:
        for lno, ln in enumerate(inf):
            dt = json.loads(ln)
            if DEBUG == True:
                assert dt["docid"] + "|||" + dt["name"] == revs_train[lno]["id"] + "|||" + revs_train[lno]["name"]
                assert dt["docid"] + "r" + "|||" + dt["name"] == revs_train[lno + len(Q_ids)]["id"] + "|||" + revs_train[lno]["name"]
    revs = revs_train + revs_test
    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


if __name__=="__main__":
    w2v_file = sys.argv[1]
    data_folder = ["data/pos.txt","data/neg.txt"]
    print "loading data...",
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=False)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    with open("max_l.txt", "w") as outf:
        outf.write(str(max_l))
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    i = 0
    datasets, trackers = make_idx_data_cv(revs, word_idx_map, i, max_l,k=300, filter_h=5)
    img_h = len(datasets[0][0])-1
    for i in range(len(datasets[1])):
        with open("test_batches/test_batches_{}.p".format(i), "w") as outf:
            cPickle.dump(datasets[1][i][:,:img_h], outf)
        if i > 0:
            datasets[1][i] = None # save space for pickle

    revs_train = [o for o in revs if o["split"] == "train"]
    print "pickling"
    with gzip.open("datasets.p", "wb") as f:
        cPickle.dump(datasets, f, -1)
    with gzip.open("trackers.p", "wb") as f:
        cPickle.dump(trackers, f, -1)
    with gzip.open("mr.p", "w") as f:
        cPickle.dump([revs_train, W, None, word_idx_map, vocab], f, -1)
    print "dataset created!"
