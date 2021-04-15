'''
Data processing helper functions.
'''

import nltk
from nltk.corpus import brown
from nltk.tokenize import casual

import json

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np

def load_msr(f, limit=None):
    '''
        Loads the MSR paraphrase corpus.
    '''
    lines = [x.strip().lower().split('\t') for x in open(f, 'r').readlines()[1:]]
    sents = [[x[3].split(), x[4].split()] for x in lines]
    labels = [int(x[0]) for x in lines]
    return sents, labels

def load_w2v(f):
    '''
        A wrapper for loading with gensim's KeyedVectors in word2vec format.
    '''
    return KeyedVectors.load_word2vec_format(f, binary=f.endswith('.bin'))

def load_kv(f):
    '''
        A wrapper for loading with gensim's KeyedVectors.
    '''
    return KeyedVectors.load(f)

def load_txt(f):
    '''
        Loads vectors from a text file.
    '''
    vectors = {}
    
    for line in open(f, 'r').readlines():
        splits = line.strip().split()
        vectors[splits[0]] = np.array([float(x) for x in splits[1:]])

    return vectors

def load_model(f):
    '''
        Guesses the file format and loads.
    '''
    if f.endswith('.bin'): return load_w2v(f)
    elif f.endswith('.txt'): return load_txt(f)
    else: return load_kv(f)
