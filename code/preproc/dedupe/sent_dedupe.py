# warning: stores all texts in memory.  need to switch to streaming sort strategy otherwise.
from collections import defaultdict
import ujson as json
import sys

seen_text_byname = defaultdict(set)
for line in sys.stdin:
    try:
        d=json.loads(line)
        text = d['sent_alter'].strip()
        if text in seen_text_byname[d['name']]: continue
        seen_text_byname[d['name']].add(text)
        print line.strip()
    except ValueError:
        print >>sys.stderr, line 
