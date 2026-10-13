#!/usr/bin/env bash
# SMS Spam Collection (Almeida & Hidalgo, UCI), as mirrored in the pycon-2016-tutorial repo.
set -euo pipefail
mkdir -p "$(dirname "$0")/data"
curl -fsSL -o "$(dirname "$0")/data/sms.tsv" \
  https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
wc -l "$(dirname "$0")/data/sms.tsv"
