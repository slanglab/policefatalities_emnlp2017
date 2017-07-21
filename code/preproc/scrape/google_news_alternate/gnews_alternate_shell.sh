#!/bin/bash
set -eu
cd "$(dirname $0)"
set +eu
(while true; do
    echo "==="
    echo "DATE $(date '+%Y-%m-%dT%H:%M:%S%z')"
    python gnews_scrape.py --police police_keywords.txt --kill kill_keywords.txt
    sleep 60
done) 2>&1 | tee -a ../logs/scraper2.log
