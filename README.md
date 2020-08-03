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

### Training

Run the code by `python3 train_textlevel_gnn.py DATASET_NAME`.

Currently support `DATASET_NAME`: R8, R52, 20ng, Ohsumed.

---
## Current Results

We are able to re-implement similar result on R8 and R52 dataset, while Ohsumed perform quite difference as compared to paper's one.

| Accuracy | R8    | R52   | 20NG  | Ohsumed | MR |
|----------|-------|-------|-------|---------|----|
| Train    | 62.6% | 56.8% | 44.8% | 16.0%   | -  |
| Valid    | 96.3% | 90.7% | 60.7% | 16.3%   | -  |
| Test     | 96.4% | 91.0% | 55.6% | 23.4%   | -  |

---
## Todo
- [x] **Implement the public edge**: Implement by adding an **UNKNOW** word node, but not sure about what the public edge really is.(In paper: *we uniformly map the edges that occur less than k times in the training set to a “public” edge to make parameters adequately trained.*)
- [ ] Result Improvement.
- [ ] Figure out why trainind accuracy is way lower than validation and testing set.
- [ ] Save model.