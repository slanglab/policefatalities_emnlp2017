import psycopg2, requests, sys, json


conn = psycopg2.connect("todo")


def check_if_archive(url):
	cur = conn.cursor()
	cur.execute("""SELECT url from wayback_archive where url=%s""", (url,))
	return cur.rowcount >= 1

def archive(url):
	try:
		print "Archiving url:"
		print url + '\n'
		req = requests.get("http://web.archive.org/save/" + url)
		cur = conn.cursor()
		headers = dict(req.headers)
		headers_json = json.dumps(headers, ensure_ascii=False) # allow non ascii symbols
		cur.execute("""INSERT INTO wayback_archive (archive_time, url, status_code, waybackheader) values (now(), %s, %s, %s)""",
					(url, req.status_code, headers_json))
		conn.commit()
		return True
	except Exception as e:
		cur = conn.cursor()
		cur.execute("""INSERT INTO wayback_archive (archive_time, url, status_code) values (now(), %s, %s)""",
					(url, 500))
		conn.commit()
		return False


def retrieve_urls(table):
	cur = conn.cursor()
	statement = """SELECT url from """ + table + """ except select url from wayback_archive"""
	cur.execute(statement)
	return cur.fetchall()


def insert_urls(table):
	urls = retrieve_urls(table)
	for url in urls:
		if not check_if_archive(url[0]):
			archive(url[0])


def main():
	# insert_urls("search_results")
	insert_urls("gnews_search_results")
	insert_urls("scraper2_search_results")
	insert_urls("wiki_killings")
	return True


main()
