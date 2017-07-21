'''
process data for CNN, round 1
'''
import json
import codecs
import sys
import os
import ipdb
from dateutil import parser
from random import randrange

FN = sys.argv[1]

with codecs.open("data/pos.txt", "w", "utf-8") as pos_outf:
    with codecs.open("data/neg.txt", "w", "utf-8") as neg_outf:
        with open(FN, "r") as inf:
            for lno, ln in enumerate(inf):
                if lno % 50000 == 0:
                    sys.stderr.write(str(lno) + ",")    
                dt = json.loads(ln.replace("\n", ""))
                plabel = dt["plabel"]
                
                # change the PERSON and TARGET tags to out of vocab words so these vectors are learned
                sent = dt["sent_alter"].replace("PERSON", "OOVPERSON123").replace("TARGET", "OOVTARGET123")
                id_ = dt["docid"]
                name = dt["name"]

                scrape_month = parser.parse(dt["downloadtime"]).month
                out = json.dumps({"lno": lno, "name": name, "doc_id": id_, "sent_alter": sent, "plabel": plabel, "scrape_month": scrape_month}) + "\n"
                if plabel == 1:
                    pos_outf.write(out)
                    out = json.loads(out)
                    # r stands for reverse. this keeps track of IDs where the label has been flipped
                    out["doc_id"] = out["doc_id"] + "r"
                    out["plabel"] = 0
                    out = json.dumps(out) + "\n"
                    neg_outf.write(out)
                elif plabel == 0:
                    neg_outf.write(out)
                    out = json.loads(out) 
                    # r stands for reverse. this keeps track of IDs where the label has been flipped
                    out["doc_id"] = out["doc_id"] + "r"
                    out["plabel"] = 1
                    out = json.dumps(out) + "\n"
                    pos_outf.write(out)
                else:
                    assert "something" == "unexpected"
