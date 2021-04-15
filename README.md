# battery

This is the code repository for HW 3 from [Natural Language Processing, Spring 2021](http://www.cs.columbia.edu/~kathy/NLP/2021/). It contains a minimal implementation of word embedding evaluation via three tasks (two intrinsic and one intrinsic), and sample solution code that generates and evaluates embedding spaces under two different algorithms (word2vec, PPMI-SVD) and parameter settings.

The writeup with full details is on Courseworks.

## Evaluation
The main script to run for evaluation is `evaluate.py`.

The three tasks used in this implementation are
* Word similarity (dataset: [WordSim353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)[1]; method: cosine similarity; evaluation metric: Spearman's rho)
* Analogy solving (dataset: [BATS](http://vecto.space/projects/BATS/)[2]; method: vector offset (3CosAdd); evaluation metric: accuracy)
* Paraphrase detection (dataset: [MSR Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)[3] via [this repo](https://github.com/wasiahmad/paraphrase_identification); method: logistic regression over cosine similarity; evaluation metric: accuracy)

See in-code documentation for more detailed instructions on how to run.

## Sample solution
Solution code is located in `solution.py`.

The solution uses `gensim` to train word2vec models and implements SVD using `scipy`'s solver. It collects co-occurrences and computes the PPMI matrix first, saves it, then loads the saved matrix for subsequent runs of SVD.

## References
[1] [Placing search in context: The concept revisited](http://www.cs.technion.ac.il/~gabr/papers/tois_context.pdf). Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. ACM Transactions on information systems, Vol. 20, No. 1, pp. 116-131. January 2002.

[2] [Analogy-based detection of morphological and semantic relations with word embeddings: what works and what doesn't](https://www.aclweb.org/anthology/N16-2002/). In Proceedings of the NAACL Student Research Workshop, pages 8–15. June 2016.

[3] [Unsupervised construction of large paraphrase corpora: Exploiting massively parallel news sources](https://www.aclweb.org/anthology/C04-1051/). COLING 2004. August 2004.