from copy import copy
import json, sys, urlparse, cgi, urllib, itertools, requests, time
from datetime import datetime,date,timedelta
import feedparser
import psycopg2
import temporal

conn 	   = psycopg2.connect("todo")
SLEEP_TIME = 10
TABLE	   = """scraper2_search_results"""

def subset(S, m):
	return set(itertools.combinations(S, m))


def keyword_combs(police_keywords, kill_keywords):
	combinations = []
	#if police_keywords:
	#	combinations.append(subset(police_keywords, 1))
	#if kill_keywords:
	#	combinations.append(subset(kill_keywords, 1))
	pairs = set()
	for pkey in police_keywords:
		for kkey in kill_keywords:
			pairs.add((pkey, kkey))
	combinations.append(pairs)
	return combinations


def build_keywords(filepath):
	f = open(filepath, 'r')
	keywords = set()
	for line in f:
		if line[0] == '#':
			continue
		else:
			keywords.add(line.strip())
	f.close()
	return keywords


def makeurl(q, relevance=False):
    args = []
    args.append(('q', q))
    args.append(('output', 'rss'))
    args.append(('ned', 'us'))
    args.append(('num', '99'))
    args.append(('tbm', 'nws'))
    
    if not relevance:
        args.append(('tbs', 'sbd:1'))
    
    qstr = urllib.urlencode(args)
    url = "https://news.google.com/news?" + qstr
    return url


def jsonsafe_feedparser_entry(entry):
    d = copy(entry)
    d['published_parsed'] = temporal.tt2str(d['published_parsed'])
    return d


def softassert(condition, msg):
    if not condition:
        print>>sys.stderr, msg


def extract_url(entry):
    links = entry['links']
    softassert(len(links)==1, "bad number of rss-specified links")
    google_url = links[0]['href']
    p = urlparse.urlparse(google_url)
    qargs = urlparse.parse_qs(p.query)

    url_ostensibly_list = qargs['url']
    softassert(len(url_ostensibly_list)==1, "bad number of url args in querystr")
    return url_ostensibly_list[0]


def check_if_exists(url, keywords):
    cur = conn.cursor()
    cur.execute("""SELECT url from """ + TABLE + """ where url=%s and keywords=%s""", (url, [keywords],))
    return cur.rowcount >= 1


# def check_if_archive(url):
# 	cur = conn.cursor()
# 	cur.execute("""SELECT url from wayback_archive where url=%s""", (url,))
# 	return cur.rowcount >= 1


# def archive(url):
# 	req = requests.post("http://web.archive.org/save/" + url)
# 	if req.status_code == 200:
# 		cur = conn.cursor()
# 		cur.execute("""INSERT INTO wayback_archive (archive_time, url) values (now(), %s)""", (url,))
# 		conn.commit()
# 		return True
# 	else:
# 		print "DID NOT ARCHIVE"
# 		print url
# 		print req.status_code

def insert_entry(entry, queryinfo, keywords):
    entry = jsonsafe_feedparser_entry(entry)
    entry_json = json.dumps(entry)
    url = extract_url(entry)

    cur = conn.cursor()

    if check_if_exists(url, keywords):
        return False

    cur.execute("""INSERT INTO """ + TABLE + """ (scrapetime,rss_id,queryinfo,url,keywords,entry) 
    				values (now(), %s, %s, %s, %s, %s)""", 
    				(entry['id'], json.dumps(queryinfo), url, [keywords], entry_json,) )
    conn.commit()
    return True


def gourl(url,keywords):
    print url
    queryinfo = {'query_url': url}

    feed = feedparser.parse(url)
    print "%d entries returned" % len(feed['entries'])
    
    ninserts = 0
    for entry in feed['entries']:
        ninserts += int(insert_entry(entry,queryinfo,keywords))
    
    print "%d added to DB" % ninserts


def arg_parse(args):
	"""
		Takes command line arguments as parameters and returns
		filepaths for police and kill keywords if given
	"""
	police_file, kill_file = False, False
	for i in range(len(args)):
		if args[i] == '--police':
			try:
				police_file = args[i + 1]
			except Exception:
				print "Incorrect format accepted parameters are: --police [file], --kill [file]"
		if args[i] == '--kill':
			try:
				kill_file = args[i + 1]
			except Exception:
				print "Incorrect format accepted parameters are: --police [file], --kill [file]"
	return police_file, kill_file


def scrape_now():
	args = sys.argv
	police_file, kill_file = arg_parse(args)
	
	police_keywords, kill_keywords = build_keywords(police_file), build_keywords(kill_file)
	key_combs = keyword_combs(police_keywords, kill_keywords)

	for m in key_combs:
		for q in m:
			url = makeurl(' '.join(q))
			gourl(url, sorted(list(q)))
			time.sleep(SLEEP_TIME)

scrape_now()
