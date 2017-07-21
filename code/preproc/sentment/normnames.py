from __future__ import division 
import string,re, argparse, json
from collections import defaultdict
from unidecode import unidecode

'''
(1) get rid of 's (or in some cases \' in s\')
(2) get rid of any names that have non-alpha symbols (make sure no "-, . symbols")
(3) get rid of titles (e.g. "Ms, Mr, Mrs etc.")
'''

def nonnum(name):
	#hyphens, periods, apostrophes need to be ok
	if re.search(u'[^a-zA-z\.\s\-\']', name)!=None: return True
	else: return False

def has_title(name):
	nsplt = set(w for w in name.lower().split())
	title_set = set(['ms.', 'mr.', 'mrs.', 'ms', 'mr', 'mrs', 'sgt.', 'sgt', 'lt', 'lt.', 'officer'])
	return len(title_set & nsplt) > 0 

def no_aposs(name):
	#strips names of 's
	#must be run before has_bad_punc
	if name[-2:] =="'s": 
		newname = name[:-2] #for examples like Bob Jone's
		return newname
	elif name[-1:]=="'": 
		newname = name[:-1]
		return newname
	else: return name

def norm(name):
	'''
	currently does all the checking with strings then will convert back to unicdoe before passing
	'''
	assert type(name) == unicode
	#next line VERY IMPORTANT to convert different types of unicode data! 
	name = unidecode(name) #converts name and all weird symbols to string ascii represntation
	assert type(name) == str
	name = no_aposs(name) #strip the apostrophe s
	if nonnum(name) : return None
	if has_title(name): return None
	return name.decode('utf-8')









