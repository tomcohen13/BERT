'''
Embedding evaluation script for COMS W4705 Spring 2021 HW3.

You can run this script from command-line on one file at a time using

    $ python evaluate.py (name-of-file-containing-trained-model) ;

or you can run it on multiple models at a time by importing the
evaluate_models() function from it and passing that a list of files.

Accepted file formats include gensim KeyedVectors files (i.e. some filepath
after you have trained a Word2Vec model and called model.wv.save(filepath)),
or .txt files with one embedding per line, where each line contains the word
and then each index of its corresponding embedding, whitespace-separated (i.e.
there should be something like

    apple 1.0 3.5 2.2 0.9 3.7

on each line).

Warning: data file paths are hardcoded in, so don't modify paths in this file
or move any of the data files in the directory it resides in!
'''

import argparse
import numpy as np
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from numpy.linalg import norm

import random
import os

from process import load_model, load_msr

def collect(model):
    '''
        Collects matrix and vocabulary list from a trained model.
        Helper function. You shouldn't have to call this yourself.
    '''
    if type(model) is dict:
        vocab = [k for k in model.keys()]
    else:
        vocab = [k for k in model.vocab.keys()]

    indices = {}
    for i in range(len(vocab)): indices[vocab[i]] = i
        
    matrix = []
    for w in vocab:
        matrix.append(model[w])
    return np.array(matrix), vocab, indices
        
def eval_wordsim(model, f='data/wordsim353/combined.tab'):
    '''
        Evaluates a trained embedding model on WordSim353 using cosine
        similarity and Spearman's rho. Returns a tuple containing
        (correlation, p-value).
    '''
    sim = []
    pred = []

    for line in open(f, 'r').readlines():
        splits = line.split('\t')
        w1 = splits[0] if splits[0] in model else splits[0].lower()
        w2 = splits[1] if splits[1] in model else splits[1].lower()
        if w1 in model and w2 in model:
            sim.append(float(splits[2]))
            v1 = model[w1]
            v2 = model[w2]
            pred.append(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

    return spearmanr(sim, pred)

def eval_bats_file(model, matrix, vocab, indices, f, repeat=False,
                   multi=0):
    '''
        Evaluates a trained embedding model on a single BATS file using either
        3CosAdd (the classic vector offset cosine method) or 3CosAvg (held-out
        averaging).

        If multi is set to zero or None, this function will usee 3CosAdd;
        otherwise it will use 3CosAvg, holding out (multi) samples at a time.

        Default behavior is to use 3CosAdd.
    '''
    pairs = [line.strip().split() for line in open(f, 'r').readlines()]

    # discard pairs that are not in our vocabulary
    pairs = [[p[0], p[1].split('/')] for p in pairs if p[0] in model]
    pairs = [[p[0], [w for w in p[1] if w in model]] for p in pairs]
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
            a = model[pairs[i][0]]
            c = model[pairs[j][0]]
            for bw in pairs[i][1]:
                qa.append(a)
                qb.append(model[bw])
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
            qb = [[w for w in pairs[j][1] if w in model] for j in range(len(pairs)) if j - i not in range(multi)]
            qbs = []
            for ws in qb: qbs += ws
            a = np.mean([model[w] for w in qa], axis=0)
            b = np.mean([np.mean([model[w] for w in ws], axis=0) for ws in qb], axis=0)
            a = a / norm(a)
            b = b / norm(b)

            for k in range(multi):
                c = model[pairs[i + k][0]]
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

def eval_bats(model, matrix, vocab, indices):
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
                accs[f.split('.')[0]] = eval_bats_file(model, matrix, vocab, indices, os.path.join(base, dr, f))
                accs[dk].append(accs[f.split('.')[0]])
            accs[dk] = [a for a in accs[dk] if a is not None]
            accs[dk] = np.mean(accs[dk]) if len(accs[dk]) > 0 else None

    accs['total'] = np.mean([accs[k] for k in accs.keys() if accs[k] is not None])

    return accs

def eval_msr(model):
    '''
        Evaluates a trained embedding model on the MSR paraphrase task using
        logistic regression over cosine similarity scores.
    '''
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

    train = [[np.sum([model[w] for w in ss[0] if w in model], axis=0), np.sum([model[w] for w in ss[1] if w in model], axis=0)] for ss in X_tr]
    test = [[np.sum([model[w] for w in ss[0] if w in model], axis=0), np.sum([model[w] for w in ss[1] if w in model], axis=0)] for ss in X_test]

    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)

def evaluate_models(files, verbose=True):
    '''
        Evaluates multiple models at a time. Returns results in a list where
        each item is a dict containing
        { "wordsim" : WordSim353 correlation,
          "bats" : a dictionary of BATS scores (see eval_bats() for details),
          "msr" : MSR paraphrase performance }.
    '''
    results = []

    for f in files:
        if verbose: print('[evaluate_models] Reading ' + f)
        model = load_model(f)
        matrix, vocab, indices = collect(model)
        r = {}
        if verbose: print('[evaluate_models] Evaluating on WordSim...')
        r['wordsim'] = eval_wordsim(model)
        if verbose: print('[evaluate_models] Evaluating on BATS...')
        r['bats'] = eval_bats(model, matrix, vocab, indices)
        if verbose: print('[evaluate_models] Evaluating on MSRPC...')
        r['msr'] = eval_msr(model)
        results.append(r)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluate a single trained model.')
    parser.add_argument('path', metavar='filename', type=str, help='the path to the file containing your trained model')
    args = parser.parse_args()

    print('[evaluate] Loading model...')
    model = load_model(args.path)

    print('[evaluate] Collecting matrix...')
    matrix, vocab, indices = collect(model)

    print('[evaluate] WordSim353 correlation:')
    ws = eval_wordsim(model)
    print(ws)

    print('[evaluate] BATS accuracies:')
    bats = eval_bats(model, matrix, vocab, indices)
    print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr(model)
    print(msr)
