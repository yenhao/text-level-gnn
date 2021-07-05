# text-level-gnn

Implementation of paper: [Text Level Graph Neural Network for Text Classification](https://www.aclweb.org/anthology/D19-1345.pdf)


---
## Dataset

Please refer to [textGCN](https://github.com/yao8839836/text_gcn/tree/master/data) and copy the **R8, R52, mr, ohsumed_single_23** to dataset folder.

For word embeddings, please refer to [GloVe](https://nlp.stanford.edu/projects/glove/). We currently support two kinds of embeddings:
1. Twitter 27B (100d & 200d): Perform better than wiki one in our implementation.
2. Wiki + Gigaword 6B (300d)

Please download these files and move to `embeddings` folder.

---
## Model 

### Settings

Separate 90% of training dataset for model training and the rest 10% for validation. The testing set is separated in the original dataset.

We keep the parameter as the same in the paper with `BATCH_SIZE = 32`, `learning rate = 0.001` with `l2 weight decay = 0.0001`.

We are also interested in the dropout settings (in model.py) as what mentioned in paper doesn't make sense. *Dropout with a keep probability of 0.5 is applied after the dense layer.* This dropout set the limitation to the label node, which could results in 0 output for all classes to the softmax function.

### Training

Run the code by `python3 train_textlevel_gnn.py DATASET_NAME`.

Currently support `DATASET_NAME`: R8, R52, 20ng, Ohsumed, MR.

---
## Current Results

We are only able to implement similar result on R8 and R52 dataset, while Ohsumed perform quite difference as compared to paper's one.

| Accuracy | R8    | R52   | 20NG  | Ohsumed | MR   |
|----------|-------|-------|-------|---------|------|
| Train    | 62.6% | 56.8% | 44.8% | 42.9%   | 69.3%|
| Valid    | 96.3% | 93.3% | 60.7% | 58.0%   | 72.0%|
| Test     | 96.4% | 91.7% | 55.6% | 54.1%   | 69.0%|

Note that the training accuracy is lower because that the author set the dropout right after dense layer.
---
## Todo
- [x] **Implement the public edge**: Implement by adding an **UNKNOW** word node, but not sure about what the public edge really is. Maybe a public edge weight? (In paper: *we uniformly map the edges that occur less than k times in the training set to a “public” edge to make parameters adequately trained.*)
- [ ] Result Improvement.
- [x] Save model.
