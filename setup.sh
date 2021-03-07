#!/bin/bash

set -e

# Set up python virtualenv
virtualenv -p python3.7 env
. env/bin/activate

# Install basic dependencies via pip
pip3 install -r requirements.txt

git submodule init
git submodule update

# Install CLIP
OLD_CWD="$(pwd)"
cd CLIP
python setup.py install
cd "$OLD_CWD"

# Install albumentations
cd albumentations
python setup.py install
cd "$OLD_CWD"

# Install torchscope
cd torchscope
python setup.py install
cd "$OLD_CWD"

# Install retinaface-pytorch
cd retinaface-pytorch
echo iglovikov-helper-functions==0.0.53 > requirements.txt
python setup.py install

# Download InsightFace-v2 ArcFace model
DOWNLOAD_CACHE_DIR="${XDG_CACHE_HOME-$HOME/.cache}/InsightFace-v2"
mkdir -p "$DOWNLOAD_CACHE_DIR"
wget -c 'https://github.com/foamliu/InsightFace-v2/releases/download/v1.0/BEST_checkpoint_r101.tar' -O "$DOWNLOAD_CACHE_DIR/BEST_checkpoint_r101.tar"
sha256sum -c <<< "c32fab2e978b25def0c201b5b5948e7068944d113438105b5b291f145393c95a  $DOWNLOAD_CACHE_DIR/BEST_checkpoint_r101.tar"

echo "All done."
