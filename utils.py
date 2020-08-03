import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize import TweetTokenizer
import multiprocessing as mp


def read_data(exp_dataset : str, data_path:str = 'dataset', dataset2folder:dict = {'R8':'R8', 'R52':'R52', '20ng':'20ng', 'Ohsumed':'ohsumed_single_23', 'MR':'mr'}):
    """Read the dataset by handling multiple datasets' structure for each experiment and return in pandas DataFrame format.

    Args:
        exp_dataset (str): The experiment dataset to conduct.
        data_path (str, optional): The folders that store the dataset. Defaults to 'dataset'.
        dataset2folder (dict, optional): The dictionary mapping for the name and folder_name of the dataset. Defaults to {'R8':'R8', 'R52':'R52', '20ng':'20ng', 'Ohsumed':'ohsumed_single_23', 'MR':'mr'}.

    Returns:
        pandas.DataFrame: The DataFrame is constructed with columns: 1.target - Specify the each row of data is for train or test purpose.
                                                                     2.label - The label for each row
                                                                     3.text - The textual content for each row
    """
    print("\nLoading dataset..")
    dataset = []
    # R20
    if exp_dataset=='R8' or exp_dataset =='R52':
        targets = ['train.txt', 'test.txt']
        for target in targets:
            text_data_path = os.path.join(data_path, dataset2folder[exp_dataset], target)
            with open(text_data_path) as f:
                lines = f.readlines()

            for line in lines:
                label, text = line.strip().split('\t')

                # add doc
                dataset.append((target[:-4], label, text))

    # ohsumed_single_23
    elif exp_dataset == 'Ohsumed':
        targets = {'training':'train', 'test':'test'}
        for target in targets:
            trainind_data_path = os.path.join(data_path, dataset2folder[exp_dataset], target)
            for label in os.listdir(trainind_data_path):
                for doc in os.listdir(os.path.join(trainind_data_path, label)):
                    with open(os.path.join(trainind_data_path, label, doc)) as f:
                        lines = f.readlines()
                    text = " ".join([line.strip() for line in lines])

                    # add doc
                    dataset.append((targets[target], label, text))

    # 20 ng
    elif exp_dataset =='20ng':
        from sklearn.datasets import fetch_20newsgroups

        for target in ['train', 'test']:
            data = fetch_20newsgroups(subset=target, shuffle=True, random_state=42, remove = ('headers', 'footers', 'quotes'))


            dataset += list(map(lambda sample: (target, data['target_names'][sample[0]], sample[1].replace("\n", " ")), zip(data['target'], data['data'])))
    # movie review
    elif exp_dataset == 'MR':
        for target in ['train', 'test']:
            text_data_path = os.path.join(data_path, dataset2folder[exp_dataset], "text_{}.txt".format(target))
            with open(text_data_path, 'rb') as f:
                text_lines = f.readlines()
            label_data_path = os.path.join(data_path, dataset2folder[exp_dataset], "label_{}.txt".format(target))
            with open(label_data_path, 'rb') as f:
                label_lines = f.readlines()
            dataset += [(target, str(label.strip()), str(text.strip())) for (text, label) in zip(text_lines, label_lines)]
    else:
        print("Wrong dataset!")
        exit()
    print("\tDataset Loaded! Total:", len(dataset))
    return pd.DataFrame(dataset, columns=["target", "label", "text"])

def data_preprocessing(data_pd:pd.DataFrame, max_len: int = 1000, tokenizer=TweetTokenizer()):
    """Tokenizing sentences/texts in the dataset and remove empty/long content.

    Args:
        data_pd (pd.DataFrame): The DataFrame of experiment dataset.
        max_len (int, optional): The maximum numbers of text tokens to keep for each data. Set `0` for keep all the tokens. Defaults to 1000.
        tokenizer (nltk.tokenize.Tokenizer(), optional): [description]. Defaults to TweetTokenizer().

    Returns:
        pandas.DataFrame: The DataFrame is constructed with an additional column `tokens` which is a set of tokens from `text` column.
    """
    print("Data Proprecessing..")
    print("\tText Tokenizing..")
    with mp.Pool(mp.cpu_count()) as p:
        data_pd['tokens'] = p.starmap(tokenize, map(lambda text: (text, tokenizer),data_pd.text.tolist()))

    print("\tRemove data with empty content")
    data_pd = data_pd[data_pd.tokens.apply(lambda tokens: (len(tokens) > 0))]
    
    # Remove the text with too many text(remove outliers)
    if max_len != 0:
        print("\t\tData trimming, max sentence length:", max_len)
        data_pd = data_pd[data_pd.tokens.apply(lambda tokens: (len(tokens) < max_len))]
        
        ## Uncomment to enable the filtering based on distribution
        # print("\n Sentence length distrbution:\n")
        # distribution = data_pd['tokens'].apply(lambda tokens: len(tokens)).describe()
        # print(distribution)

        # MAX_LEN = int(distribution['mean']+ 3*distribution['std'])
        # print("\t\tRemove text > {} tokens (mean + 3x std of word counts)".format(MAX_LEN))

        # data_pd = data_pd[data_pd.tokens.apply(lambda tokens: len(tokens) < distribution['mean'] + 3 * distribution['std'])]

        print("\n\t\tFinish trimming, Total Rest:", len(data_pd))

    return data_pd

def tokenize(text:str, tokenizer):
    return [w.lower() for w in tokenizer.tokenize(text)]

def features_extracting(data_pd: pd.DataFrame, minimum_word_count: int = 15, neighbor_distance: int = 2) :
    """Extract words, word's neighbor features as well as transform the label as label index.

    Args:
        data_pd (pd.DataFrame): [description]
        minimum_word_count (int, optional): [description]. Defaults to 15.
        neighbor_distance (int, optional): [description]. Defaults to 2.

    Returns:
        pandas.DataFrame: The DataFrame is constructed with an additional column `X`, `NX`, `y` for word's index, neighbor word information and label index.
        dict: The dictionary that mapping word to index  
    """
    print("\n Feature Extracting..")
    word2idx, unknow_idx = construct_word2idx(data_pd.tokens.to_list(), minimum_word_count= minimum_word_count)

    # get index representation of words
    # data_pd['X'] = data_pd.tokens.apply(lambda tokens: [word2idx.get(w) if word2idx.get(w) else unknow_idx for w in tokens])
    data_pd['X'] = transform_word2idx_mp(data_pd.tokens.to_list(), word2idx=word2idx)

    # get word's neighbors with specific neighbor distance
    data_pd['X_Neighbors'] = get_word_neighbors_mp(data_pd.X.to_list(), neighbor_distance=neighbor_distance)

    # get label by it category index
    data_pd['y'] = pd.Categorical(data_pd.label).codes

    print("\tFinish, adding `X`, `X_Neighbors` and `y` columns to DataFrame")
    return data_pd, word2idx

def construct_word2idx(tokens_of_texts: list, minimum_word_count: int = 15) :
    """[summary]

    Args:
        tokens_of_texts (list): A list of the tokens of sentences/texts from dataset.
        minimum_word_count (int, optional): The lowerbound of the wordcount, the word with frequency lower than it would be discard. Defaults to 15.

    Returns:
        dict: The dictionary that map word to its index, where index `0` is assigned to the padding `_PAD_` and last index for `_UNKNOW_` words (rare word as well).
        int: The index of `_UNKNOW_` word.
    """
    word_counts = Counter([w for tokens in tokens_of_texts for w in tokens])
    print("\tMost common words:", word_counts.most_common(10))
    # remove rare words
    qualified_words = [w for w,v in word_counts.items() if v > minimum_word_count]
    print("\tTotal words:", len(qualified_words))


    word2idx = {word:i+1 for i,word in enumerate(qualified_words)}
    word2idx[0] = '_PAD_'
    word2idx[len(qualified_words)] = '_UNKNOW_'
    return word2idx, len(qualified_words)

def transform_word2idx_mp(tokens_of_texts: list, word2idx: dict) :
    """Multi processing version of converting a set of word token to it's index

    Args:
        tokens_of_texts (list): A list of the tokens of sentences/texts from dataset.
        word2idx (dict): The dictionary that map word to its index, where index `0` is the padding `_PAD_` and last index for unknow.

    Returns:
        list: A list of indice of the tokens from sentence.
    """
    print("\tConverting word to it's index")
    with mp.Pool(4) as p:
        return p.starmap(transform_word2idx, map(lambda tokens: (tokens, word2idx), tokens_of_texts))

def transform_word2idx(tokens: list, word2idx: dict) :
    """Converting a set of word token to it's index

    Args:
        tokens (list): A list of the tokens of sentences/texts from dataset.
        word2idx (dict): The dictionary that map word to its index, where index `0` is the padding `_PAD_` and last index for unknow.

    Returns:
        list: A list of indice of the tokens from sentence.
    """
    unknow_idx = len(word2idx)-1
    return [word2idx.get(w) if word2idx.get(w) else unknow_idx for w in tokens]

def get_word_neighbors_mp(text_tokens:list, neighbor_distance:int) :
    print("\tGet word's neighbors")
    with mp.Pool(mp.cpu_count()) as p:
        return p.starmap(get_word_neighbor, map(lambda tokens: (tokens, neighbor_distance), text_tokens))

def get_word_neighbor(text_tokens: list, neighbor_distance: int) :
    """Get word token's adjacency neighbors with distance : neighbor_distance

    Args:
        text_tokens (list): A list of the tokens of sentences/texts from dataset.
        neighbor_distance (int): The adjacency distance to consider as a neighbor.

    Returns:
        list: A nested list with 2 dimensions, which is a list of neighbor word tokens (2nd dim) for all tokens (1nd dim)
    """
    text_len = len(text_tokens)

    edge_neighbors = []
    for w_idx in range(text_len):
        skip_neighbors = []
        # check before
        for sk_i in range(neighbor_distance):
            before_idx = w_idx -1 - sk_i
            skip_neighbors.append(text_tokens[before_idx] if before_idx > -1 else 0)

        # check after
        for sk_i in range(neighbor_distance):
            after_idx = w_idx +1 +sk_i
            skip_neighbors.append(text_tokens[after_idx] if after_idx < text_len else 0)

        edge_neighbors.append(skip_neighbors)
    return edge_neighbors
