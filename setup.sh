#!/bin/bash

set -eu

# Set up python virtualenv
virtualenv -p python3.7 env
. env/bin/activate

# Install basic dependencies via pip
pip3 install -r requirements.txt

git submodule init
git submodule update

# Install faiss for fast similarity search
OLD_CWD="$(pwd)"
cd faiss
perl -pi -e 's!cmake_minimum_required\(VERSION 3.17 FATAL_ERROR\)!cmake_minimum_required(VERSION 3.16 FATAL_ERROR)!' CMakeLists.txt ./faiss/python/CMakeLists.txt
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make -C build/ -j4 faiss
make -C build/ -j4 swigfaiss
cd build/faiss/python && python setup.py install
cd "$OLD_CWD"

# Install CLIP
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
mkdir -p ~/.cache/InsightFace-v2
wget -c 'https://github.com/foamliu/InsightFace-v2/releases/download/v1.0/BEST_checkpoint_r101.tar' -O ~/.cache/InsightFace-v2/BEST_checkpoint_r101.tar
sha256sum -c <<< "c32fab2e978b25def0c201b5b5948e7068944d113438105b5b291f145393c95a  $HOME/.cache/InsightFace-v2/BEST_checkpoint_r101.tar"
