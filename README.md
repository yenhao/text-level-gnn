# text-level-gnn

Implementation of paper: [Text Level Graph Neural Network for Text Classification](https://www.aclweb.org/anthology/D19-1345.pdf)


---
## Dataset

Please refer to [textGCN](https://github.com/yao8839836/text_gcn/tree/master/data) and copy the **R8, R52, mr, ohsumed_single_23** to dataset folder.

For word embeddings, please refer to [GloVe](https://nlp.stanford.edu/projects/glove/). We currently support two kinds of embeddings:
1. Twitter 27B (100d & 200d)
2. Wiki + Gigaword 6B (300d)


---
## Model Training

Run the code by `python3 train_textlevel_gnn.py DATASET_NAME`.

Currently support `DATASET_NAME`: R8, R52, 20ng, Ohsumed.


---
## Todo
1. Implement the public edge: not sure about what the public edge is.(Besides, we uniformly map the edges that occur less than k times in the training set to a “public” edge to make parameters adequately trained.)
2. Result Improvement.
3. Save model.