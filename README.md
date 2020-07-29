# text-level-gnn

Implementation of paper: [Text Level Graph Neural Network for Text Classification](https://www.aclweb.org/anthology/D19-1345.pdf)


---
## Dataset

Please refer to [textGCN](https://github.com/yao8839836/text_gcn/tree/master/data) and copy the **R8, R52, mr, ohsumed_single_23** to dataset folder.

For word embeddings, please refer to [GloVe](https://nlp.stanford.edu/projects/glove/). We currently support two kinds of embeddings:
1. Twitter 27B (100d & 200d)
2. Wiki + Gigaword 6B (300d)

Please download these files and move to `embeddings` folder.

---
## Model 

### Settings

Separate 90% of training dataset for model training and the rest 10% for validation. The testing set is separated in the original dataset.

We keep the parameter as the same in the paper with `BATCH_SIZE = 32`, `learning rate = 0.001` with `l2 weight decay = 0.0001`.

### Training

Run the code by `python3 train_textlevel_gnn.py DATASET_NAME`.

Currently support `DATASET_NAME`: R8, R52, 20ng, Ohsumed.

---
## Current Results


R8 (8 cls)
Acc: 62.6%(train)   96.3%(valid)    96.4%(test)

R52 (52 cls)
Acc: 56.8%(train)   90.7%(valid)    91.0%(test)

20ng (20 cls)
Acc: 44.8%(train)   60.7%(valid)    55.6%(test)

Ohsumed (23 cls)
Acc: 16.0%(train)   16.3%(valid)    23.4%(test)


---
## Todo
1. **Implement the public edge**: not sure about what the public edge is.(Besides, we uniformly map the edges that occur less than k times in the training set to a “public” edge to make parameters adequately trained.)
2. Result Improvement.
3. Save model.