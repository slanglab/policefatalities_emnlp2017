from __future__ import division
from nltk import ngrams
from collections import defaultdict
import ujson as json
from sklearn.feature_extraction import FeatureHasher
import argparse
from scipy import sparse, io

#use https://github.com/brendano/stanford_corenlp_pywrapper
from stanford_corenlp_pywrapper import CoreNLP
cc=CoreNLP(annotators="tokenize,ssplit,pos,depparse") 

#example to manually change path of corenlp 
#cc=CoreNLP(annotators="tokenize,ssplit,pos,depparse", corenlp_jars=["/Users/KatieKeith/stanford-corenlp-full-2016-10-31/*"])


#--------------BELOW IS FOR N-GRAMS---------
def get_all_ngrams(tok_idxs):
    ng = []
    for i in [1, 2, 3]:
        for n in ngrams(tok_idxs, i): ng.append(n)
    return ng

def a1(ng, tokens):
    if len(ng) == 1: return tokens[ng[0]]
    else:
        s = ''
        for i in range(len(ng)-1):
            s+= tokens[ng[i]]+u','
        s+=tokens[ng[-1]]
        return s

def a2(ng, tokens, pos_tags):
    if len(ng) == 1: return tokens[ng[0]]+','+pos_tags[ng[0]]
    else:
        s = ''
        for i in range(len(ng)-1):
            s+= tokens[ng[i]]+','+pos_tags[ng[i]]+','
        s+=tokens[ng[-1]]+','+pos_tags[ng[-1]]
        return s

def a3(ng, tokens, targ_idxs):
    result = []
    for t in targ_idxs:
        if len(ng) == 1: 
            result.append(tokens[ng[0]]+'_'+str(ng[0]-t))
        else:
            s = ''
            for i in range(len(ng)-1):
                s+= tokens[ng[i]]+'_'+str(ng[i]-t)+','
            s+=tokens[ng[-1]]+'_'+str(ng[-1]-t)
            result.append(s)
    return result

def get_sides_targ(tokens, targ_idxs):
    #get two on either side of target or close enough
    result = []
    for t in targ_idxs:
        start = t - 2
        end = t + 2
        if start < 0: start = 0 
        if end >= len(tokens): end = len(tokens) - 1
        result.append((start, end))
    return result 

def a4(sides_targ, pos_tags):
    s = ''
    for i in range(sides_targ[0], sides_targ[1]+1):
        s+=pos_tags[i]+','
    return s.strip(',')

def a5(sides_targ, tokens, pos_tags):
    s = ''
    for i in range(sides_targ[0], sides_targ[1]+1):
        s+=tokens[i]+','+pos_tags[i]+','
    return s.strip(',')


#--------------BELOW IS FOR DEPENDENCIES ------------
def get_edges_dir(deps):
    #takes core nlp deps and returns edges dict and direc 
    edges = defaultdict(set)
    direc = {}
    deptups = {}
    for dep in deps: 
        if dep[1]==-1 or dep[2]==-1: continue 
        edges[dep[1]].add(dep[2])
        edges[dep[2]].add(dep[1])   
        direc[(dep[1], dep[2])] = u'>'+dep[0]
        direc[(dep[2], dep[1])] = u'<'+dep[0]
    return edges, direc

def paths(node, edges, visited, path=(), length=1):
    #gets all the paths of specified length 
    #node: you will have to add the node you start with to this
    #reuturns tuples of idexes 
    if length==1:
        for neigh in edges[node]:
            if neigh in visited: continue
            yield path+(node, neigh)
    else:
        visited.add(node)
        for neigh in edges[node]:
            if neigh in visited: continue 
            for result in paths(neigh, edges, visited, path, length=length-1):
                yield result

#b1-b3
def get_paths_incl_targ(edges, targ_idxs):
    #gets all the paths that include one of the targ indexs 
    start_nodes =  get_start_nodes(edges, targ_idxs)
    allpaths = [] #all paths with the target
    #length denotes the number of edges between 
    for strt in start_nodes:
        for p in paths(strt, edges, set(), (), length=2):
            path = (strt, )+p
            for targ in targ_idxs: 
                if targ in path: 
                    allpaths.append(path)
    return allpaths

def b1(path, tokens, direc, pos_tags):
    #changes the output from paths() into unicode 
    s = u''
    for i in range(len(path)-1):
        s+=tokens[path[i]]+','+pos_tags[path[i]]+','+direc[(path[i], path[i+1])]+','
    s+=tokens[path[-1]]+','+pos_tags[path[-1]]
    return s

def b2(path, tokens, direc):
#changes the output from paths() into unicode 
    s = u''
    for i in range(len(path)-1):
        s+=tokens[path[i]]+u','+direc[(path[i], path[i+1])]+u','
    s+=tokens[path[-1]]
    return s

def b3(path, tokens, pos_tags):
    #changes the output from paths() into unicode 
    s = u''
    for i in range(len(path)-1):
        s+=tokens[path[i]]+u','+pos_tags[path[i]]+u','
    s+=tokens[path[-1]]+u','+pos_tags[path[-1]]
    return s

#b4 (will need to send thru b1)
def get_len_2(direc):
    return direc.keys()

def get_start_nodes(edges, targ_idxs):
    #this is needed to every starting 3-length dep that will have TARGET
    start_nodes = set()
    for t in targ_idxs:
        start_nodes.add(t)
        neighs = edges[t]
        start_nodes = start_nodes.union(neighs)
        for n in neighs: 
            start_nodes = start_nodes.union(edges[n])
    return start_nodes

def extr_all_feats(filename, output_file, hasNgrams=True, hasDeps=True):
    allfeats = []
    doc_count = 0 
    zero_feats = 0
    with open(filename, 'r') as r:
        for line in r:
            doc_count += 1
            sent = json.loads(line)['sent_alter']
            #sometimes TARGET and PERSON gets weird merge with other characters 
            sent = sent.replace('TARGET', ' TARGET ')
            sent = sent.replace('PERSON', ' PERSON ')
            assert type(sent) == unicode 
            d = cc.parse_doc(sent)
            feats = defaultdict(float)

            for s in d['sentences']:
                SYMBOLS = set("TARGET PERSON".split())
                tokens=[w.lower() if w not in SYMBOLS else w for w in s['tokens']]
                if 'TARGET' not in tokens: continue
                assert u'TARGET' in tokens 
                deps = s['deps_cc']
                pos_tags = s['pos']
                targ_idxs =[i for i, x in enumerate(tokens) if x == 'TARGET'] #there can be multiple TARGETS in a sentence
                tok_idxs = [i for i in range(len(tokens))]

                #-----EXTRACT FEATURES-------
                if hasNgrams:
                    for ng in get_all_ngrams(tok_idxs):
                        feats[a1(ng, tokens)] += 1.0
                        feats[a2(ng, tokens, pos_tags)] += 1.0
                        for f in a3(ng, tokens, targ_idxs): #multiple for multiple TARGET indexs
                            feats[f] += 1.0
                    for sides_targ in get_sides_targ(tokens, targ_idxs):
                        feats[a4(sides_targ, pos_tags)] += 1.0
                        feats[a5(sides_targ, tokens, pos_tags)] += 1.0
                
                if hasDeps:
                    edges, direc = get_edges_dir(deps)
                    #getting b1 thru b3 feats
                    for path in get_paths_incl_targ(edges, targ_idxs):
                        feats[b1(path, tokens, direc, pos_tags)] += 1.0
                        feats[b2(path, tokens, direc)] += 1.0
                        feats[b3(path, tokens, pos_tags)] += 1.0

                    #getting b4 feats
                    for path in get_len_2(direc):
                        feats[b1(path, tokens, direc, pos_tags)] += 1.0
            if len(feats) == 0: zero_feats+= 1
            allfeats.append(feats)
    assert len(allfeats) == doc_count
    print "READ {0} DOCS FROM FILE {1}".format(doc_count, filename)
    print "NUM DOCS WITH ZERO FEATS=",zero_feats
    output_file = 'feats_'+output_file+'.json'
    w = open(output_file, 'w')
    json.dump(allfeats, w)
    print 'wrote allfeats to ', output_file
    return allfeats

def go_feathash(feats, output_file, max_feats=125000, save=True):
    #feature hasher for training
    print "hasing to {0} features".format(max_feats)
    fh = FeatureHasher(n_features=max_feats)
    X = fh.transform(feats)
    assert X.shape[1] == max_feats
    #X.shape[0]==doc_count
    if save: 
        io.mmwrite(output_file, X)
        print "feat matrix saved to '{0}'.mtx".format(output_file)
    else: return X

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ngrams', action='store_true', help='includes A feats, ngram-like feats')
    arg_parser.add_argument('--deps', action='store_true', help='includes B feats, dep-like feats')
    arg_parser.add_argument('input', type=str, help='input file')
    args = arg_parser.parse_args()

    output_file = args.input.split('/')[-1].split('.')[0]
    if args.ngrams: output_file+= '_'+'ng'
    if args.deps: output_file += '_'+'dep'

    print "NGRAMS={0}, DEPS={1}".format(args.ngrams, args.deps)
    MAX_FEATS = 450000 #change based on the dimension you wish to feature hash to 
    allfeats = extr_all_feats(args.input, output_file, hasNgrams=args.ngrams, hasDeps=args.deps)
    go_feathash(allfeats, output_file, max_feats=MAX_FEATS)






    













