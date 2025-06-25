#!/bin/bash

set -e

echo "📥 Downloading COCO 2017 train images..."
wget -c http://images.cocodataset.org/zips/train2017.zip

echo "📥 Downloading COCO 2017 val images..."
wget -c http://images.cocodataset.org/zips/val2017.zip

echo "📦 Unzipping train2017.zip..."
unzip -q train2017.zip

echo "📦 Unzipping val2017.zip..."
unzip -q val2017.zip

echo "🧹 Removing image ZIPs..."
rm train2017.zip val2017.zip

echo "📥 Downloading annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "📦 Unzipping annotations..."
unzip -q annotations_trainval2017.zip

echo "🧹 Removing annotations ZIP..."
rm annotations_trainval2017.zip

echo "Finished downloading dataset"
