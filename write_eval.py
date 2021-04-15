import os
import sys

from evaluate import evaluate_models
from evalbert import evaluate_bert

# Eval csv location
CSV_NAME = 'models_eval.csv'
CSV_NAME_BERT = 'bert_eval.csv'
SVD_LOC = 'svd'
W2V_LOC = 'word2vec'

# BATS categories
bats_cat_1 = "lexicographic_semantics"
bats_cat_2 = "E10 [male - female]"
bats_cat_3 = "total"

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f'Usage: {sys.argv[0]} ["all" or "concrete_model_path" or "bert"]')
        return

    all_model_paths = []

    if sys.argv[1] == 'all':
        w2v_names = [model for model in os.listdir(W2V_LOC) if 'word2vec' in model]
        svd_names = [model for model in os.listdir(SVD_LOC) if 'svd' in model]
        w2v_paths = [os.path.join(W2V_LOC, model) for model in w2v_names]
        svd_paths = [os.path.join(SVD_LOC, model) for model in svd_names]

        all_model_names = w2v_names + svd_names
        all_model_paths = w2v_paths + svd_paths

    elif sys.argv[1] == 'bert':
        res = evaluate_bert()
        wordsim, bats1, bats2, bats3, msr = extract_scores(res)

        with open(CSV_NAME_BERT, 'w') as f:
            line = 'WordSim, BATS 1, BATS 2, BATS 3, MSR\n'
            print(line[:-1])
            f.write(line)

            line = f'{wordsim}, {bats1}, {bats2}, {bats3}, {msr}\n'
            print(line[:-1])
            f.write(line)

        return

    else:
        all_model_paths.append(sys.argv[1])

    all_model_names = [os.path.basename(m_path) for m_path in all_model_paths]

    res = evaluate_models(all_model_paths)

    csv_exists = os.path.exists(CSV_NAME)

    with open(CSV_NAME, 'a') as f:

        for i, model in enumerate(all_model_names):

            if '.model' in model:
                model = model[:-6]

            if '.txt' in model:
                model = model[:-4]

            model_spec = model.split('_')

            name = model_spec[0]
            w = model_spec[1][2:]
            dim = model_spec[2][2:]
            ns = '-'
            if name == 'word2vec':
                ns = model_spec[3][2:]

            res_model = res[i]
            wordsim, bats1, bats2, bats3, msr = extract_scores(res_model)

            if not csv_exists:
                f.write('Algorithm, Win., Dim., N.s., WordSim, BATS 1, BATS 2, BATS 3, MSR\n')
                csv_exists = True

            f.write(f'{name}, {w}, {dim}, {ns}, {wordsim}, {bats1}, {bats2}, {bats3}, {msr}\n')

def extract_scores(res_model):
    wordsim = round(100*res_model["wordsim"].correlation,2)
    msr = round(100*res_model["msr"],2)

    bats1 = round(res_model["bats"][bats_cat_1],3)
    bats2 = round(res_model["bats"][bats_cat_2],3)
    bats3 = round(res_model["bats"][bats_cat_3],3)

    return wordsim, bats1, bats2, bats3, msr


if __name__ == '__main__':
    main()
