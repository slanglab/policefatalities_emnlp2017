### Description of each downloader 

##### Archiver (submits URLs to the wayback machine).
   - see archiving/archive_shell.sh
   - reads from the database URLs from wayback_archive and sends them to wayback machine.

##### Google news finder (505 queries).
   - see google_news/google_shell.sh
   - runs a keyword search in Google news. There are 505 queries from across 2 lists on this downloader.
   - High level process: (1) create the URL for google news query (2) get the RSS feed (3) parse the RSS (4) gets the entries from RSS (5) extract info from RSS feeds and puts in DB.
   - If 2 queries get to same URL, then it goes in the DB twice.
   - However, if a query/URL pair is already in the DB, it will skip it.
   - Do not exceed 1 query to Google news every 6 seconds. Rotate 1 of the 505 API calls every 6 seconds.

##### Page downloader
   - page_download/downloader.sh
   - Downloads the raw HTML for all of the URLs we have collected from Google news queries
   - goes thru the DB looking for URLS that have not get been downloaded
   - It looks at 4 tables: fe_killings, wiki_killings, gnews_search_results, scraper2_search_results
   - It puts the data into page/downloads


#### More notes on the google news scraper

Google News scraper

The scraper queries Google News regularly throughout the day with the same set
of queries, and if it finds new articles, it saves them to the database.

The parameters for google news RSS API calls are somewhat documented here:
http://i-tweak.blogspot.com/2013/10/google-news-search-parameters-missing.html
Of course, this page may not be perfect or perfectly up to date.

We use these parameters:

The search query and RSS output format
    args.append(('q', q))
    args.append(('output', 'rss'))

This is what Google News calls the "regional edition", which we set to "U.S.
(English)." This seems to somewhat restrict to news about the U.S., though
it's certainly not always the case for all articles that get selected.
    args.append(('ned', 'us'))

Ask for a large number (99) results instead of the default
    args.append(('num', '99'))

Restrict to "only news."  Google News seems to have a notion of more blog-y
things that this excludes.  Not super sure what it means, but it seemed to get
better data.
    args.append(('tbm', 'nws'))

Use date-based, not relevance-based, ranking.
    args.append(('tbs', 'sbd:1'))

