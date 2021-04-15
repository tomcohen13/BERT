'''
Embedding evaluation script for COMS W4705 Spring 2021 HW3.

This script has been modified to use BERT, specifically BERT base model (uncased).

You can run this script from command-line using

    $ python evalbert.py

or you can run it from a script by importing evaluate_bert().

Warning: data file paths are hardcoded in, so don't modify paths in this file
or move any of the data files in the directory it resides in!
'''

import numpy as np
import torch
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from numpy.linalg import norm

import random
import os
import sys
import pickle as pkl

from transformers import BertTokenizerFast, BertModel

from process import load_msr

name = f'[{sys.argv[0][:-3]}]'

def model_out(s, model, tokenizer):
    '''
        Takes in a string, bert model and tokenizer,
        returns a numpy array of the mean of hidden states
        without the cls and the sep tokens.
    '''
    enc = tokenizer(s, return_tensors='pt')
    out = model(**enc)
    bef_sep = out.last_hidden_state.shape[1] - 1
    hs_no_cls_sep = out.last_hidden_state[:,1:bef_sep,:]
    ret = torch.mean(hs_no_cls_sep, 1).squeeze().detach().numpy()
    return ret

def collect(model, tokenizer):
    '''
        Collects matrix and vocabulary list from a trained model.
        Helper function. You shouldn't have to call this yourself.
    '''
    indices = tokenizer.vocab

    inverse_vocab = {i:w for w,i in indices.items()}

    vocab = []
    for i in range(len(indices)):
        vocab.append(inverse_vocab[i])

    matrix = []
    try:
        print(f'{name} Attempting to load BERT vocab/embedding matrix...')
        with open('bert_matrix.pkl', 'rb') as f:
            matrix = pkl.load(f)

    except OSError:
        print(f'{name} Failed. Rebuilding BERT vocab/embedding matrix...')

        for i, w in enumerate(vocab):
            embed = model_out(w, model, tokenizer)
            matrix.append(embed)
            if i % 500 == 0:
                print(f'{name} Processing word: {i}/{len(vocab)}')

        print(f'{name} Saving...')
        with open('bert_matrix.pkl', 'wb') as f:
            pkl.dump(matrix, f)

    return np.array(matrix), vocab, indices
        
def eval_wordsim(model, tokenizer, f='data/wordsim353/combined.tab'):
    '''
        Evaluates a trained embedding model on WordSim353 using cosine
        similarity and Spearman's rho. Returns a tuple containing
        (correlation, p-value).
    '''
    header_passed = False
    sim = []
    pred = []

    for line in open(f, 'r').readlines():
        if not header_passed:
            header_passed = True
            continue
        splits = line.split('\t')
        w1 = splits[0]
        w2 = splits[1]
        v1 = model_out(w1, model, tokenizer)
        v2 = model_out(w2, model, tokenizer)
        sim.append(float(splits[2]))
        pred.append(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

    return spearmanr(sim, pred)

def eval_bats_file(model, tokenizer, matrix, vocab, indices, f, repeat=False,
                   multi=0):
    '''
        Evaluates a trained embedding model on a single BATS file using either
        3CosAdd (the classic vector offset cosine method) or 3CosAvg (held-out
        averaging).

        If multi is set to zero or None, this function will usee 3CosAdd;
        otherwise it will use 3CosAvg, holding out (multi) samples at a time.

        Default behavior is to use 3CosAdd.
    '''
    cached_out = lambda w: matrix[indices[w]] if w in indices else model_out(w, model, tokenizer)
    pairs = [line.strip().split() for line in open(f, 'r').readlines()]

    # discard pairs that are not in our vocabulary
    pairs = [[p[0], p[1].split('/')] for p in pairs if p[0] in indices]
    pairs = [[p[0], [w for w in p[1] if w in indices]] for p in pairs]
    pairs = [p for p in pairs if len(p[1]) > 0]
    if len(pairs) <= 1: return None

    transposed = np.transpose(np.array([x / norm(x) for x in matrix]))

    if not multi:
        qa = []
        qb = []
        qc = []
        targets = []
        exclude = []
        groups = []
        
        for i in range(len(pairs)):
            j = random.randint(0, len(pairs) - 2)
            if j >= i: j += 1
            a = cached_out(pairs[i][0])
            c = cached_out(pairs[j][0])
            for bw in pairs[i][1]:
                qa.append(a)
                qb.append(cached_out(bw))
                qc.append(c)
                groups.append(i)
                targets.append(pairs[j][1])
                exclude.append([pairs[i][0], bw, pairs[j][0]])

        for queries in [qa, qb, qc]:
            queries = np.array([x / norm(x) for x in queries])
        
        sa = np.matmul(qa, transposed) + .0001
        sb = np.matmul(qb, transposed)
        sc = np.matmul(qc, transposed)
        sims = sb + sc - sa

        # exclude original query words from candidates
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    else:
        offsets = []
        exclude = []
        preds = []
        targets = []
        groups = []
        
        for i in range(len(pairs) // multi):
            qa = [pairs[j][0] for j in range(len(pairs)) if j - i not in range(multi)]
            qb = [[w for w in pairs[j][1] if w in indices] for j in range(len(pairs)) if j - i not in range(multi)]
            qbs = []
            for ws in qb: qbs += ws
            a = np.mean([cached_out(w) for w in qa], axis=0)
            b = np.mean([np.mean([cached_out(w) for w in ws], axis=0) for ws in qb], axis=0)
            a = a / norm(a)
            b = b / norm(b)

            for k in range(multi):
                c = cached_out(pairs[i + k][0])
                c = c / norm(c)
                offset = b + c - a
                offsets.append(offset / norm(offset))
                targets.append(pairs[i + k][1])
                exclude.append(qa + qbs + [pairs[i + k][0]])
                groups.append(len(groups))

        print(np.shape(transposed))

        sims = np.matmul(np.array(offsets), transposed)
        print(np.shape(sims))
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    preds = [vocab[np.argmax(x)] for x in sims]
    accs = [1 if preds[i].lower() in targets[i] else 0 for i in range(len(preds))]
    regrouped = np.zeros(np.max(groups) + 1)
    for a, g in zip(accs, groups):
        regrouped[g] = max(a, regrouped[g])
    return np.mean(regrouped)

def eval_bats(model, tokenizer, matrix, vocab, indices):
    '''
        Evaluates a trained embedding model on BATS.

        Returns a dictionary containing
        { category : accuracy score over the category }, where "category" can
        be
            - any of the low-level category names (i.e. the prefix of any of
              the individual data files)
            - one of the four top-level categories ("inflectional_morphology",
              "derivational_morphology", "encyclopedic_semantics",
              "lexicographic_semantics")
            - "total", for the overall score on the entire corpus
    '''
    accs = {}
    base = 'data/BATS'
    for dr in os.listdir('data/BATS'):
        if os.path.isdir(os.path.join(base, dr)):
            dk = dr.split('_', 1)[1].lower()
            accs[dk] = []
            for f in os.listdir(os.path.join(base, dr)):
                accs[f.split('.')[0]] = eval_bats_file(model, tokenizer, matrix, vocab, indices, os.path.join(base, dr, f))
                accs[dk].append(accs[f.split('.')[0]])
            accs[dk] = [a for a in accs[dk] if a is not None]
            accs[dk] = np.mean(accs[dk]) if len(accs[dk]) > 0 else None

    accs['total'] = np.mean([accs[k] for k in accs.keys() if accs[k] is not None])

    return accs

def eval_msr(model, tokenizer):
    '''
        Evaluates a trained embedding model on the MSR paraphrase task using
        logistic regression over cosine similarity scores.
    '''
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')
    out = lambda seq: model_out(" ".join(seq), model, tokenizer)

    train = [[out(ss[0]), out(ss[1])] for ss in X_tr]
    test = [[out(ss[0]), out(ss[1])] for ss in X_test]

    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)

def evaluate_bert(verbert = 'bert-base-uncased', verbose=True):
    '''
        Evaluates BERT (default bert-base-uncased). Returns results in a
        dict containing
        { "wordsim" : WordSim353 correlation,
          "bats" : a dictionary of BATS scores (see eval_bats() for details),
          "msr" : MSR paraphrase performance }.
    '''
    print(f'[evaluate_bert] Loading model {verbert}...')
    tokenizer = BertTokenizerFast.from_pretrained(verbert)
    model = BertModel.from_pretrained(verbert)
    res = {}

    print(f'[evaluate_bert] Collecting matrix...')
    matrix, vocab, indices = collect(model, tokenizer)

    if verbose: print('[evaluate_bert] Evaluating on WordSim...')
    res['wordsim'] = eval_wordsim(model, tokenizer)
    if verbose: print('[evaluate_bert] Evaluating on BATS...')
    res['bats'] = eval_bats(model, tokenizer, matrix, vocab, indices)
    if verbose: print('[evaluate_bert] Evaluating on MSRPC...')
    res['msr'] = eval_msr(model, tokenizer)

    return res

if __name__ == "__main__":
    verbert = 'bert-base-uncased'

    if len(sys.argv) == 2:
        verbert = 'bert-base-uncased'

    print(f'{name} Loading model {verbert}...')
    tokenizer = BertTokenizerFast.from_pretrained(verbert)
    model = BertModel.from_pretrained(verbert)

    print(f'{name} Collecting matrix...')
    matrix, vocab, indices = collect(model, tokenizer)

    print(f'{name} WordSim353 correlation:')
    ws = eval_wordsim(model, tokenizer)
    print(ws)
 
    print(f'[{sys.argv[0][:-3]}] BATS accuracies:')
    bats = eval_bats(model, tokenizer, matrix, vocab, indices)
    print(bats)

    print(f'[{sys.argv[0][:-3]}] MSR accuracy:')
    msr = eval_msr(model, tokenizer)
    print(msr)
