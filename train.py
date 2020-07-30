#!/usr/bin/env python
# coding: utf-8
import sys
import os
import utils
import model


# get experiment dataset
if len(sys.argv) != 2:
    print("Run as : CUDA_VISIBLE_DEVICES=0 python3 train_textlevel_gnn.py DATASET_NAME")
    exit()

experiment_dataset = sys.argv[1]

print("Experiment on :", experiment_dataset) #, sys.argv)

### HYPER PARAMETERS
MIN_WORD_COUNT = 15  # the word with frequency less than this num will be remove
NEIGHBOR_DISTANCE = 2  # same as paper's one
WORD_EMBED_DIM = 300 # dimension for word embedding
PRETRAIN_EMBEDDING = True # whether to use pretrain or not
MODEL_MAX_SEQ_LEN = 0  # the length of text should the model encode/learning, set 0 to consider all

N_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOP_EPOCHS = 20  # after n_epochs not improve then stop training
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
NUM_NODES = len(word2idx) # with padding and unknow noes
NUM_CLASSES = len(set(data_pd['y'].values))

# Construct model
text_level_gnn = model.TextLevelGNN_Model(NUM_NODES, NUM_CLASSES, WORD_EMBED_DIM, PRETRAIN_EMBEDDING, MODEL_MAX_SEQ_LEN)

# Assign experiment dataset
text_level_gnn.set_dataset(data_pd)

# Training and Evaluation
text_level_gnn.train_eval(N_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_EPOCHS, EARLY_STOP_MONITOR)




"""
R8 (8 cls)
Acc: 62.6%(train)   96.3%(valid)    96.4%(test)

R52 (52 cls)
Acc: 56.8%(train)   90.7%(valid)    91.0%(test)

20ng (20 cls)
Acc: 44.8%(train)   60.7%(valid)    55.6%(test)

Ohsumed (23 cls)
Acc: 16.0%(train)   16.3%(valid)    23.4%(test)
"""

