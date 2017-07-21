#!/bin/bash
set -eu
cd "$(dirname $0)"
set +eu
(while true; do
    echo "==="
    echo "DATE $(date '+%Y-%m-%dT%H:%M:%S%z')"
    python archive.py
    sleep 60
done) 2>&1 | tee -a ../logs/archive.log
