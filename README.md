# Text-HGCN

The implementation of Text-HGCN in our paper

## Require

Python 3.6

Torch 1.10.0

DGL 0.9.1

## Reproducing Results

1. Run `python 1_data_processing.py 20ng`

2. Run `python 2_word2vec.py 20ng`

3. Run `python 3_doc2vec.py 20ng`

4. Run `python 4_training_labels.py 20ng`

5. Run `python 5_edge_word2doc.py 20ng`

6. Run `python 6_edge_word2word.py 20ng`

7. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.
