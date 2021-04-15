hw3
Adam Storek and Tom Cohen
as5827 tc2955

Q1: training battery of word2vec and SVD

Implemented exactly as specified in the lab.
Consisting of 2 python scripts:

train.py	main routine to train the models

	$ python train.py all		trains all models*
	$ python train.py word2vec	trains word2vec models*
	$ python train.py svd		trains svd models*

	*across all the hyperparameters listed in the hw spec, as applicable
	to each architecture

	Trains the models and then saves them on the disk in the following
	formats in their respective folders:

	Word2Vec: KeyedVectors binary
	"word2vec/word2vec_w={window_size}_d={dimension}_k={n.s.}.model

	SVD: .txt file as specified by evaluate.py
	"svd/svd_w={window_size}_d={dimension}.txt

	Details of the implementation:

	For Word2Vec, smoothing p=0.75 is used as in the original paper.
	Furthemore, since my laptop has 12 vcpus, I used 12 threads.

	For SVD, we found that creating the co-occurence matrix in a dense
	matrix performed best and was not consuming too much memory. Computing
	PPMI proved to be a bigger challenge, and an elegant vectorized
	solution used too much memory. In our implementation, therefore, we
	use double for loop to iterate over each entry, since we found it to
	be  the most memory efficient, even compared to using different sparse
	matrices (coo_matrix, lil_matrix, csr_matrix) and this compression
	on-the-fly did not work well. The construction of the PPMI matrix took
	~20 minutes on a laptop with 16GB of RAM. The PPMI matrix is saved on
	the disk and the script checks for it each time it attempts to train
	on a given window size, therefore the PPMI building cost is
	considerably amortized.

write_eval.py	main routine to evaluate all the models

	$ python write_eval.py all	evaluates all models (w2v and svd)
	$ python write_eval.py word2vec	evaluates w2v models
	$ python write_eval.py svd	evaluates svd models
	$ python write_eval.py bert	evaluates bert (described further on)

	Calls evaluate_models and also builds a csv file models_eval.csv to
	ease the table building in Q3, using global-parameter-controlled bats
	categories and rounding.

Q2

Implemented exactly as specified in the lab.
Consisting of 2 scripts:

evalbert.py	main routine to evaluate BERT, essentially bert-adapted
		evaluate.py

		Uses Hugging Face's bert-base-uncased model by default. It
		generally gets the BERT embedding for a word/sequence by
		taking the mean of the hidden states except the CLS token and
		the SEP token hidden states. For most words (those in BERTs
		vocab), this means simply taking the one hidden state.
		However, for unknown words (tokenized into subwords) and
		sentences, taking the mean becomes meaningful.

		For BATS, I extract BERT's vocab from its tokenizer and
		compute the embeddings for each word. This can take up to ~20
		mins on my laptop (wish I had CUDA...), but again I save the
		resulting matrix on the disk, and the script looks for it next
		time, so the cost is amortized to some extent. Furthermore,
		MSR can also take 20-30 mins on a CPU (another one from the
		series "wish I had CUDA...").

write_eval.py	Not necessary to evaluate BERT, but can be used to also
		generate a CSV used in 3.: bert_eval.csv

		$ python write_eval.py bert	evaluates bert using the same
						rounding, bats categories as
						set for Q1 models
