#!/bin/bash
set -eu
cd "$(dirname $0)"
set +eu
(while true; do
    echo "==="
    echo "DATE $(date '+%Y-%m-%dT%H:%M:%S%z')"
    python consolidated_page_downloader.py
    sleep 600
done) 2>&1 | tee -a ../logs/download.log
