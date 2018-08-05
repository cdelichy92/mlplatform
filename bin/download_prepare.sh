#!/bin/bash

# bash script to download datasets and word vectors

# download imdb reviews dataset
wget -P ../mlplatform/datasets/ https://github.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/raw/master/imdb_tr.csv

# preprocess dataset to be in specified format
source activate mlplatform
python preprocess_datasets.py

# download glove word vectors (300D, 42B tokens from common crawl)
wget -P word_vectors/ http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip ../mlplatform/word_vectors/gloved.42B.300d.zip
rm ../mlplatform/word_vectors/gloved.42B.300d.zip

wget -P ../mlplatform/word_vectors/ http://nlp.stanford.edu/data/glove.6B.zip
unzip ../mlplatform/word_vectors/glove.6B.zip
rm ../mlplatform/word_vectors/glove.6B.zip
