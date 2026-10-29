#!/usr/bin/env bash
set -euo pipefail

BASE="http://images.cocodataset.org"
FILES=(
  "zips/train2017.zip"
  "zips/val2017.zip"
  "zips/test2017.zip"
  "annotations/annotations_trainval2017.zip"
)

for f in "${FILES[@]}"; do
  (
    z="$(basename "$f")"
    aria2c -x16 -s16 -k1M -c --console-log-level=warn --summary-interval=0 -o "$z" "$BASE/$f"
    unzip -q -o "$z"
    rm "$z"
  ) &
done
wait
