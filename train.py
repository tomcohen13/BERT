import os
import sys
import pickle

# word2vec dependencies
from gensim.models.word2vec import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# svd dependencies
from scipy import sparse
import numpy as np

# Global parameters
THREADS = 12
CORPUS_PATH = 'data/brown.txt'

paramlist_window = (2, 5, 10)
paramlist_dimension = (50, 100, 300)
paramlist_k = (1, 5, 15)

name = f'[{sys.argv[0][:-3]}]'

class BrownCorpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r') as f:
            for line in f:
                words = [w.lower() for w in line.split() if BrownCorpus.is_valid_token(w)]
                yield words

    def is_valid_token(token):
        if len(token) > 2 and ('-' in token or "'s" in token):
            return True
        return token.isalpha()

brown = BrownCorpus('data/brown.txt')

class SVDModel:
    def __init__(self, dimension):
        self.dim = dimension
        self.W = None
        self.word_dict = None

    def build_ppmi_matrix(corpus, window):
        print(f'{name} Building co-occurrence matrix.')
        words = []
        lines_total = 0
        for line in corpus:
            lines_total += 1
            words.extend(line)

        word_set = set(words)
        n_uniq_w = len(word_set)
        words = list(word_set)
        words.sort()

        word_dict = dict(zip(words, list(range(n_uniq_w))))

        # Possible to use a dense matrix for the Brown vocab
        # 16GB RAM is enough - and much faster than using sparse
        cooc_mat = np.zeros((n_uniq_w, n_uniq_w), dtype=np.uint32)

        for line_no, line in enumerate(corpus):
            line_len = len(line)

            for i,word in enumerate(line):
                start = max(i - window,0)
                end = min(i + window + 1, line_len)
                context = line[start:i] + line[i+1:end]

                for context_word in context:
                    cooc_mat[word_dict[word]][word_dict[context_word]] += 1

            if line_no % 5000 == 0:
                print(f'{name} Processing lines: {line_no}/{lines_total}')

        print(f'{name} Co-occurrence matrix built.')

        print(f'{name} Computing PPMI...')
        ppmi_mat = SVDModel.cooc2ppmi(cooc_mat)
        print(f'{name} Done.')

        print(f'{name} Transforming dense PPMI matrix to coo_matrix to use for training.')
        coo_ppmi_mat = sparse.coo_matrix(ppmi_mat)

        # Probably don't want to store 19GB on disk (incl. word_dict)
        print(f'{name} Transforming sparse PPMI matrix to lil_matrix to save disk space.')
        lil_ppmi = coo_ppmi_mat.tolil()

        path = f'svd/ppmi_mats/ppmi_mat_w={window}.pkl'
        print(f'{name} Dumping (lil_ppmi, word_dict) to disk: {path}')
        with open(path, 'wb') as sf:
            pickle.dump((lil_ppmi, word_dict), sf)

        print(f'{name} Done.')

        return coo_ppmi_mat, word_dict

    def cooc2ppmi(cooc_mat):
        sum_w = np.sum(cooc_mat, 0)
        sum_c = np.sum(cooc_mat, 1)
        sum_total = np.sum(sum_c)
        p_i = sum_w/sum_total
        p_j = sum_c/sum_total
        p_ij = cooc_mat/sum_total
        ppmi = np.zeros((len(p_i),len(p_j)), dtype=np.float32)
        for i in range(len(p_i)):
            for j in range(len(p_j)):
                if p_i[i] == 0 or p_j[j] == 0:
                    # Word only appears in its own context - case of one word lines
                    continue
                pre_log = p_ij[i,j]/(p_i[i]*p_j[j])
                if pre_log > 1:
                    ppmi[i,j] = np.log2(pre_log)
            if (i % 500) == 0:
                print(f'{name} Processing rows: {i}/{len(p_i)}')
        return ppmi

    def train(self, ppmi_mat, word_dict):
        print(f'{name} Training model...')
        u,s,v = sparse.linalg.svds(ppmi_mat.asfptype(), k = self.dim)
        self.W = u @ np.sqrt(np.diag(s))
        self.word_dict = word_dict
        print(f'{name} Training finished.')

    def save(self, path):
        print(f'{name} Saving model...')
        with open(path, 'w') as f:
            for word, index in self.word_dict.items():
                line = np.array2string(self.W[index], max_line_width = 100000000).strip("[[]]")
                f.write(f'{word} {line}\n')
        print(f'{name} Model saved.')

def run_word2vec():
    print(f'{name} Training word2vec battery...')
    if not os.path.exists('word2vec') or not os.path.isdir('word2vec'):
        os.mkdir('word2vec')

    sg = 1 # skipgram enable
    ns_exponent = 0.75 # smoothing as in the original Word2Vec paper

    for window in paramlist_window:
        for dimension in paramlist_dimension:
            for k in paramlist_k:
                model = Word2Vec(sentences = brown,
                                 window = window,
                                 workers = THREADS,
                                 sg = sg,
                                 negative = k,
                                 size = dimension,
                                 ns_exponent = ns_exponent)
                model.wv.save(f'word2vec/word2vec_w={window}_d={dimension}_k={k}.model')

def run_svd():
    print(f'{name} Training svd battery...')
    if not os.path.exists('svd') or not os.path.isdir('svd'):
        os.mkdir('svd')

    if not os.path.exists('svd/ppmi_mats') or not os.path.isdir('svd/ppmi_mats'):
        os.mkdir('svd/ppmi_mats')

    for window in paramlist_window:

        ppmi_mat = None
        word_dict = None

        try:
            print(f'{name} Attempting to load a PPMI matrix from the disk')
            with open(f'svd/ppmi_mats/ppmi_mat_w={window}.pkl', 'rb') as f:
                lil_ppmi, word_dict = pickle.load(f)
                ppmi_mat = sparse.coo_matrix(lil_ppmi)

        except OSError:
            print(f'{name} PPMI matrix not found, rebuilding...')
            ppmi_mat, word_dict = SVDModel.build_ppmi_matrix(brown, window)

        for dimension in paramlist_dimension:
            SVDm = SVDModel(dimension)
            SVDm.train(ppmi_mat, word_dict)
            SVDm.save(f'svd/svd_w={window}_d={dimension}.txt')

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print(f'Usage: {sys.argv[0]} ["all", "word2vec", or "svd"]')
        return

    if sys.argv[1] == 'all':
        run_word2vec()
        run_svd()

    if sys.argv[1] == 'word2vec':
        run_word2vec()

    if sys.argv[1] == 'svd':
        run_svd()

if __name__ == '__main__':
    main()
