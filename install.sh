#!/bin/bash



#module load python/3.10
#module load arrow/16.1.0
which python3

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install nltk nlpaug numpy pandas

python3 -m pip install urllib3 tabulate
python3 -m pip install ./downloads/libsource/transformers
python3 -m pip install scipy==1.10.1 dill
python3 -m pip install accelerate -U
python3 -m pip install scikit-learn
python3 -m pip install gensim
python3 -m pip install datasets evaluate
python3 -m pip install urllib3 boto3
python3 -m pip install deepspeed
python3 -m pip install matplotlib
pip install -U pip setuptools wheel
python3 -m pip install spacy
python -m spacy download en_core_web_sm
python3 -m pip install psutil
mkdir saves
mkdir saves/models
mkdir saves/tmp
mkdir downloads
mkdir downloads/libsource
mkdir downloads/libsourcev2
