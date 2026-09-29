#!/bin/bash
# Package a finished AWS stack for upload:  bash pack.sh <name> <s2_artifact_name>
#   e.g. bash pack.sh v6rc3 s2_v6_rich_ce_ce2_ce3
set -e
NAME=$1; S2=$2
cd "C:/Users/Aman/Desktop/Amazon-ML"
mkdir -p submissions/$NAME
scp -q er:~/er/sub_$NAME/matching_results.tsv submissions/$NAME/
scp -q "er:~/er/code/business_entity_resolution/models/${S2}.*" "er:~/er/code/business_entity_resolution/models/${NAME}_config.json" code/business_entity_resolution/models/
md5sum submissions/$NAME/matching_results.tsv
ssh er "md5sum ~/er/sub_$NAME/matching_results.tsv"
python make_package.py Embedding > /dev/null
mv dist/Embedding_code.zip submissions/${NAME}_Embedding_code.zip
PYTHONIOENCODING=utf-8 python check_submission.py submissions/$NAME submissions/${NAME}_Embedding_code.zip --zip-only | tail -1
ls -la submissions/${NAME}_Embedding_code.zip
df -h /c | tail -1
