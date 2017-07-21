
from __future__ import division
from __future__ import absolute_import
import json, argparse, re
import normnames
from collections import defaultdict
import sys
from hap.hapnorm import get_hap_to_org
from unidecode import unidecode 

'''
input:: cleaned, sentence segmented docs

this code does: 
(1) name normalization
(3) get rid of all one token names  
(2) HAPNIS names
(3) checks name against gold standard 
(4) writes new sent with TARGET and PERSON symbols 

output :: normalized names and altered sentences written
'''

def get_sent_alter(targets, persons, sent_org):
    sent_alter = unidecode(sent_org)
    for target in targets: 
        target= target.replace('-', ' - ')
        sent_alter = sent_alter.replace(target, ' TARGET ')
    for person in persons: 
        sent_alter = sent_alter.replace(person, ' PERSON ')
    return sent_alter

def read_gold(gold_f):
    '''
    reads gold .tsv file
    outputs unicode tuple (date, name, city)
    '''
    gold = {}
    with open(gold_f, 'r') as r:
        for line in r:
            date = json.loads(line)["date"]
            name = json.loads(line)["name"]
            city = json.loads(line)["city"]
            assert type(date) == type(name) == type(city) == unicode
            gold[name.lower()] = {"city": city, "date": date}

    print "READ GOLD FILE: ", gold_f
    return gold 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="filed with scraped docs after cleanup (usually in abe's text_extraction_abe/data/results", type=str)
    parser.add_argument("gold", help=".json file with gold standard (name, date, location)", type=str)
    args = parser.parse_args()

    gold_dict = read_gold(args.gold)
    gold_set = set(gold_dict.keys())
    assert type(gold_set) == set

    name2docid = defaultdict(list)
    docid2obj = dict()

    with open(args.input, 'r') as r: 
        for line in r:
            obj_in = json.loads(line) 
            names = [n["string"].strip() for n in obj_in["people"]]
            sent_org = obj_in["raw"]
            docid = obj_in["id"]
            downloadtime = obj_in["downloadtime"]

            sentnames = []

            for name_org in names:
                name_replace = name_org.replace('\n', ' ')
                name_replace = normnames.norm(name_replace) #this also runs thru unidecode 
                if name_replace == None: continue #bad names that come from normnames
                if len(name_replace.split()) < 2: continue #one token names 
                sentnames.append(name_replace)
                name2docid[name_replace].append(docid)

            obj= {"sent_org": sent_org, 
                   "docid": docid,
                   "downloadtime": downloadtime,
                   "sentnames": sentnames
                  }
            docid2obj[docid] = obj

    #check name_norm against the gold database
    tmp = args.input[-2:]
    hapnames = get_hap_to_org(name2docid.keys(), tmp)
    
    #-----BOTH MATCHING RULES------
    predict_no = 'results/{0}.out'.format(args.input[-2:])
    w_no = open(predict_no, 'w')

    predict_nc = 'results_nc/{0}.out'.format(args.input[-2:])
    w_nc = open(predict_nc, 'w')
    
    for hapname, targ_names in hapnames.iteritems():
        if len(hapname.split()) < 2: continue #no one-token names! 
        targ_docid= []
        for t in targ_names:
            targ_docid += [d for d in name2docid[t]]
        targ_docid = set(targ_docid)

        for d in targ_docid: 
            obj = docid2obj[d]
            namesinsent = obj["sentnames"]
            sent_alter = get_sent_alter(set(targ_names), set(namesinsent)-set(targ_names), obj["sent_org"])

            #this gets rid of any other weid 's in the middle of names or tother things we haven't found yet 
            if re.search('TARGET', sent_alter) == None: continue 
            assert re.search('TARGET', sent_alter) != None

            obj["sent_alter"] = sent_alter

            #positives 
            if hapname.lower() in gold_set:
                #NAME ONLY MATCHES
                obj["plabel"] = 1
                obj["name"] = hapname
                obj["names_org"] = targ_names 
                obj["incidentdate"] = gold_dict[hapname.lower()]["date"]
                json.dump(obj, w_no)
                w_no.write('\n')  

                #NAME CITY MATCHES
                if re.search(gold_dict[hapname.lower()]["city"], obj["sent_alter"], re.IGNORECASE) !=None: 
                    obj["city"] = gold_dict[hapname.lower()]["city"]
                    json.dump(obj, w_nc)
                    w_nc.write('\n')
                        
            #negatives
            else:
                obj["plabel"] = 0
                obj["name"] = hapname
                obj["names_org"] = targ_names 
                #writing twice to each#file but oh well, most likely will only use name city file 
                json.dump(obj, w_nc)
                json.dump(obj, w_no)
                w_nc.write('\n')
                w_no.write('\n')

    w_no.close()
    w_nc.close()
    print "wrote NAME-ONLY data to %s" % predict_no
    print "wrote NAME-CITY data to %s" % predict_nc




    
