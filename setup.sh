#!/bin/bash

# Set up python virtualenv
virtualenv -p python3.7 env
. env/bin/activate

# Install basic dependencies via pip
pip3 install -r requirements.txt

# Install faiss for fast similarity search
OLD_CWD="$(pwd)"
git clone https://github.com/facebookresearch/faiss
cd faiss
perl -pi -e 's!cmake_minimum_required\(VERSION 3.17 FATAL_ERROR\)!cmake_minimum_required(VERSION 3.16 FATAL_ERROR)!' CMakeLists.txt ./faiss/python/CMakeLists.txt
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make -C build/ -j4 faiss
make -C build/ -j4 swigfaiss
cd build/faiss/python && python setup.py install
cd "$OLD_CWD"

# Install CLIP
git clone https://github.com/openai/CLIP
cd CLIP
python setup.py install
