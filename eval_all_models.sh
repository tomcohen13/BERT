#!/bin/bash

for model in word2vec/*.model; do
	python write_eval.py $model
done

for model in svd/*.txt; do
	python write_eval.py $model
done
