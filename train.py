#!/usr/bin/env python
# coding: utf-8
import sys
import utils
import model


# get experiment dataset
if len(sys.argv) != 2:
    print("Arguments error!\nPlease run as : CUDA_VISIBLE_DEVICES=0 python3 train.py DATASET_NAME")
    exit()

experiment_dataset = sys.argv[1]

print("Experiment on :", experiment_dataset) #, sys.argv)

### HYPER PARAMETERS
MIN_WORD_COUNT = 15  # the word with frequency less than this num will be remove and consider as unknown word later on (probabily is the k mention in paper)
NEIGHBOR_DISTANCE = 2  # same as paper's one

WORD_EMBED_DIM = 200 # dimension for word embedding
PRETRAIN_EMBEDDING = True  # use pretrain embedding or not
PRETRAIN_EMBEDDING_FIX = False # skip the training for pretrain embedding or not
MODEL_MAX_SEQ_LEN = 0  # the length of text should the model encode/learning, set 0 to consider all

N_EPOCHS = 1000
WARMUP_EPOCHS = 250  # VERY important to reproduce their result in this implementation..

## Training params from paper
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOP_EPOCHS = 50  # after n_epochs not improve then stop training
EARLY_STOP_MONITOR = "loss" # monitor early stop on validation's loss or accuracy.
###

# Read dataset
data_pd = utils.read_data(experiment_dataset)

# Data preprocessing
data_pd = utils.data_preprocessing(data_pd)

# Feature Extraction
data_pd, word2idx = utils.features_extracting(data_pd, minimum_word_count=MIN_WORD_COUNT, neighbor_distance=NEIGHBOR_DISTANCE)

print("\n", data_pd.head())


# # Model
NUM_CLASSES = len(set(data_pd['y'].values))

# Construct model
text_level_gnn = model.TextLevelGNN_Model(word2idx, NUM_CLASSES, WORD_EMBED_DIM, PRETRAIN_EMBEDDING, PRETRAIN_EMBEDDING_FIX, MODEL_MAX_SEQ_LEN)

# Assign experiment dataset
text_level_gnn.set_dataset(data_pd)

# Training and Evaluation
text_level_gnn.train_eval(N_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_EPOCHS, EARLY_STOP_MONITOR, WARMUP_EPOCHS)




"""
R8 (8 cls) [embedding_dim=200, early_stop_monitor="loss", at epoch:235]
Acc: 63.1%(train)   97.8%(valid)    96.6%(test)

R52 (52 cls) [embedding_dim=200, early_stop_monitor="loss", at epoch:324]
Acc: 55.6%(train)   93.3%(valid)    91.7%(test)

20ng (20 cls) [embedding_dim=200, early_stop_monitor="accuracy"]
Acc: 47.3%(train)       64.2%(valid)    57.2%(test)

Ohsumed (23 cls) [embedding_dim=200]
Acc: 42.9%(train)       58.0%(valid)    54.1%(test)
"""

