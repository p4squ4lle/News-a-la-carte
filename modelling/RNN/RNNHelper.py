import random
import numpy as np
import pandas as pd

def import_behaviors(path_to_file):    
        fn = path_to_file.split("/")[-1]
        assert "behaviors" in fn, f"file does not seem to be the behaviors file (path: {path_to_file})"
            
        behaviors = pd.read_csv(path_to_file)
        behaviors['history_split'] = behaviors.history.str.split(' ')
        behaviors['impressions_split'] = behaviors.impressions.str.split(' ')
        
        return behaviors

def dataframe_to_numpy(dataframe, return_columndict=True):    
    df_array = dataframe.to_numpy(copy=True)
    columndict = {name: idx for idx, name in enumerate(dataframe.columns)}
    
    if return_columndict:
        return df_array, columndict
    
    return df_array

def encode_articles(dataframe):

    df_array, columndict = dataframe_to_numpy(dataframe)
    
    print("Creating list of all articles in behaviors...")
    articles = []
    for row in df_array:
        for article in row[columndict['history_split']]:
            articles.append(article)    
        for article in row[columndict['impressions_split']]:
            articles.append(article[:-2]) 
    
    print("Creating unique articles set")
    unique_articles = set(articles)
    num_articles = len(unique_articles)
    article2idx = {u:i for i, u in enumerate(unique_articles)}
    
    print("Encoding articles in dataframe with integers...")
    dataframe['history_int'] = dataframe.history_split.apply(lambda x: [article2idx[i] for i in x])
    
    dataframe['impressions_int_1'] = dataframe.impressions_split.apply(lambda x: [article2idx[art[:-2]] for art in x if art[-1] == '1'])
    
    dataframe['impressions_int_0'] = dataframe.impressions_split.apply(lambda x: [article2idx[art[:-2]] for art in x if art[-1] == '0'])
    
    return dataframe, unique_articles, num_articles, article2idx

def create_pos_neg(dataframe, n_hist_articles=5, npratio=1):
    
    impressions_1 = dataframe["impressions_int_1"].to_numpy()
    impressions_0 = dataframe["impressions_int_0"].to_numpy()
    articles_as_int = dataframe['history_int'].to_numpy()

    assert len(articles_as_int) == len(impressions_0) == len(impressions_1), "something wrong!"
    
    print("Create complete list of 'positive' articles...")
    complete_list_1s = []
    for i, hist in enumerate(articles_as_int):
        list_1s = []
        for art_int in hist[-n_hist_articles:]:
            list_1s.append(art_int)
        list_1s.append(impressions_1[i][0])
        complete_list_1s.append(list_1s)
    
    assert len(complete_list_1s) == len(articles_as_int), "oh no!"
   
    print("Create complete list of 'negative' articles...")
    complete_list_0s = []
    for i, hist in enumerate(articles_as_int):
        list_0s = []
        for art_int in hist[-n_hist_articles:]:
            list_0s.append(art_int)
        
        negs = impressions_0[i]
        if npratio > len(negs):
            neg = random.sample(negs*(npratio//len(negs)+1), npratio)
        else:
            neg = random.sample(negs, npratio)
        
        for n in neg:
            complete_list_0s.append(list_0s + [n])
    
    assert len(complete_list_1s) == len(complete_list_0s) / npratio, "almost did it, but still something wrong"
    
    return complete_list_1s, complete_list_0s

def rnn_train_val_split(complete_list_1s, complete_list_0s, train_ratio=0.8, val_ratio=0.2, random_seed=420):
    
    assert train_ratio + val_ratio == 1, f"incosistent train and val ratios ({train_ratio}, {val_ratio})"
    
    number_of_indexes = len(complete_list_1s)
    npratio = len(complete_list_0s)//len(complete_list_1s)
    
    random.seed(random_seed)
    train_indexes = random.sample(range(number_of_indexes), int(train_ratio*number_of_indexes))
    test_indexes = list(set(range(number_of_indexes)) - set(train_indexes))
    
    complete_list_1s_train = [complete_list_1s[i] for i in train_indexes]
    complete_list_1s_test = [complete_list_1s[i] for i in test_indexes]
    
    complete_list_0s_train = []
    for i in train_indexes:
        for t in range(npratio):
            train_list = complete_list_0s[i*npratio + t]
            complete_list_0s_train.append(train_list)
    
    assert len(complete_list_1s_train) == len(complete_list_0s_train) / npratio, "something wrong with the inputs"
    
    complete_list_0s_test = []
    for i in test_indexes:
        for t in range(npratio):
            test_list = complete_list_0s[i*npratio + t]
            complete_list_0s_test.append(test_list)
    
    assert len(complete_list_1s_test) == len(complete_list_0s_test) / npratio, "something wrong with the inputs"
    
    train = []
    train_targets = []
    for i, l in enumerate(complete_list_1s_train):
        train.append(l)
        train_targets.append(1)
        for j in range(npratio):
            train.append(complete_list_0s_train[i*npratio + j])
            train_targets.append(0)
    
    valid = []
    valid_targets = []
    for i, l in enumerate(complete_list_1s_test):
        valid.append(l)
        valid_targets.append(1)
        for j in range(npratio):
            valid.append(complete_list_0s_test[i*npratio + j])
            valid_targets.append(0)
    
    assert len(train_targets) == len(train)
    assert len(valid_targets) == len(valid)

    train_array = np.array(train)
    valid_array = np.array(valid)
    train_targets_array = np.array(train_targets)
    valid_targets_array = np.array(valid_targets)
    
    train_array = train_array.reshape(len(train), 6)
    valid_array = valid_array.reshape(len(valid), 6)

    return (train_array, valid_array, 
            train_targets_array, valid_targets_array,
            train_indexes, test_indexes)