import requests
import json
import psycopg2
import sys
import logging
import os
from random import shuffle
from requests.structures import CaseInsensitiveDict

LOG_PATH = "download.log"

try:
    os.remove(LOG_PATH) # remove the last download log if exists
except OSError:
    pass

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

filehandler = logging.FileHandler(LOG_PATH)

# Create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s - %(funcName)s - ' +
    '%(levelname)s - %(lineno)d - %(message)s')
filehandler.setFormatter(formatter)

# Add the handlers to the logger
log.addHandler(filehandler)

conn_str = ""
conn = psycopg2.connect(conn_str)


def get_search_results(table, source):
    cur = conn.cursor()
    statement = "SELECT url from " + table +\
                " except (select url from " +\
                "page_downloads where source='%s')" % source
    cur.execute(statement)
    result = cur.fetchall()
    return result


def unicode_safe_header(header_dict):
    safe_key_values = [(key, unicode(value, errors='ignore'))
                       for key, value in header_dict.iteritems()]
    return CaseInsensitiveDict(safe_key_values)


def get_response(url):
    try:
        response = requests.get(url, verify=False)
        status = response.status_code
        safe_header = unicode_safe_header(response.headers)
        header = json.dumps(dict(safe_header))
        content = response.content
    except (requests.exceptions.ConnectionError,
            requests.exceptions.ContentDecodingError,
            requests.exceptions.TooManyRedirects) as e:
        status = 500
        header = ''
        content = ''
        log.debug("requests.exceptions.ConnectionError|ContentDecodingError|TooManyRedirects {}".format(url))
        print>>sys.stderr, e
    except (requests.exceptions.MissingSchema):
        log.debug("requests.exceptions.MissingSchema {}".format(url))
        return get_response('http://' + url)
    return status, header, content


def insert_entry(url, source):
    if url == '':
        return False
    print "Downloading Page:"
    print url, "\n"

    status, header, content = get_response(url)
    cur = conn.cursor()
    cur.execute("""INSERT into page_downloads
                   (url, downloadtime, status_code,
                    http_header, page_content, source)
                    values (%s, now(), %s, %s, %s, %s)""",
                (url, status, header,
                 psycopg2.Binary(content), source,))
    conn.commit()
    return True


def duplicate_check(url, source):
    cur = conn.cursor()
    cur.execute(
        "SELECT url from page_downloads where url=%s and source=%s",
        (url, source,))
    return cur.rowcount >= 1


if __name__ == '__main__':
    count = 0

    # Google News page downloading
    # This scraper is no longer running because new scraper2
    # captures all the same queries and provides keyword data

    # ids = get_search_results_url()
    # print len(ids)
    # for i in ids:
    # 	ID, url = i[0], i[1]
    # 	count += int(google_insert_entry(ID, url))

    ######################################################################
    ######################################################################

    # Fatal Encounters page downloading
    source = 'FE'
    fe_ids = get_search_results('fe_killings', source)
    for url in fe_ids:
        count += int(insert_entry(url[0], source))

    ######################################################################
    ######################################################################

    # Wikipedia Page Downloading
    source = 'WIKI'
    wiki_ids = get_search_results('wiki_killings', source)
    for url in wiki_ids:
        count += int(insert_entry(url[0], source))

    ######################################################################
    ######################################################################

    # scraper2; Original queries new scraper
    source = 'SCRAPER2'
    scraper2_ids = get_search_results('scraper2_search_results', source)
    for url in scraper2_ids:
        log.info("trying {}".format(url))
        count += int(insert_entry(url[0], source))

    log.info("Number of pages downloaded %d" % count)

    ######################################################################
    ######################################################################

    # GNews scraper
    source = 'GNEWS'
    gnews_ids = get_search_results('gnews_search_results', source)
    shuffle(gnews_ids)
    for url in gnews_ids:
        log.info("trying {}".format(url))
        count += int(insert_entry(url[0], source))
