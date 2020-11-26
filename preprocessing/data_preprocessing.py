#!/usr/bin/env python
# coding: utf-8

# # Preprocessing and Cleaning of the Data

import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp

from progressbar import ProgressBar

pickle_matrix = False
# ## Choose whether to load the small or large dataset

# Choose from "small" or "large"
dataset_size = "large"
# Choose from "train" or "dev" (test)
dataset_type = "train"

dataset_path = f"../../data/mind_{dataset_size}_{dataset_type}/"
behaviors_path = dataset_path + "behaviors.tsv"
news_path = dataset_path + "news.tsv"


# ## Loading the data

behaviors = pd.read_csv(behaviors_path, sep='\t', header=None)
news = pd.read_csv(news_path, sep='\t', header=None)


# Let's first give our datasets some proper column names:

behaviors = behaviors.rename(columns={0: 'impression_id', 
                                      1: 'user_id', 
                                      2: 'time', 
                                      3: 'history', 
                                      4: 'impressions'})

news = news.rename(columns={0: 'article_id', 
                            1: 'category', 
                            2: 'subcategory', 
                            3: 'title', 
                            4: 'abstract', 
                            5: 'url', 
                            6: 'title_entities', 
                            7: 'abstract_entities'})

news_shape = news.shape
print(f"\nShape of news dataset: {news_shape}")
print(f"There are more than {news_shape[0]//1000},000 news articles in our news dataset.")

# For every article we have information concerning the **news category, subcategory, it's title, abstract and even some entitiy embeddings** (most of the urls don't work anymore so we don't have access to the full bodies). Let's check whether these are all unique articles or if we also have some duplicates:

print("Number of unique news articles: ", news.title.nunique())
print("Number of duplicates:             ", news.shape[0] - news.title.nunique())


# Apparently, there are **news articles with multiple IDs**. We don't just want to drop them yet, as this would result in a loss of useful information concerning the click behaviors and reading histories in our ***behaviors* dataset**, which looks like this:

# Let us first prepare this dataset before we get back to handling the duplicate news articles.

# ## Preparing the *behaviors* dataset

behaviors_shape = behaviors.shape
print(f"\nShape of behaviors dataset: {behaviors_shape}")
print(f"In the behaviors dataset there are more than {behaviors_shape[0]//1000},000",
      "online sessions from MSN news.")


# We have information concerning **user ID, date and daytime, the click history, and the recommended articles and user behavior** (ending on -1 = clicked) for the respective session. Let's see how many unique users there are:

print(f'There are {len(behaviors.user_id.unique())} individual users in our dataset.')
print(f'The average number of sessions is: {behaviors_shape[0] / len(behaviors.user_id.unique()):.1f}')

# we want to include **only users with at least five articles read** in their history. So we need to drop ahistorical users as well as users with too few articles read:

behaviors.dropna(inplace=True)

behaviors['length_history'] = behaviors.history.str.split().map(len)

behaviors = behaviors[behaviors['length_history'] >= 5]

# ## Droppping duplicate article IDs in *news* and remapping them in *behaviors*
# Now that we have set up our *behaviors* dataset with respect to users, let's get back to the duplicate news articles. With different IDs for the de facto same articles we would not be able to track similarities among users sufficiently. In the following, we will **replace every redundant article-ID with the first ID for the respective article**. In order to this, we first create a subset with all the duplicates in it:

duplis_title = news[news.duplicated(subset="title", keep=False)]

duplis_title.sort_values(by="title").head(3)

title_set = duplis_title['title'].unique()

article_list = []
for title in title_set:
    x = duplis_title[duplis_title['title']==title]['article_id'].to_list()
    article_list.append(x)

# We now have a **list which contains article IDs for every article which has multiple IDs** in our original dataset. With this list, we can generate a dictionary called articleID_dict, which maps all the redundant IDs (keys) to a single ID (value):

articleID_dict = {}
articles_to_change = []
for article in article_list:
    value = article[0]
    keys = article [1:]
    for k in keys:
        articleID_dict[k] = value
        articles_to_change.append(k)


# Let's make a copy of the original behaviors dataframe and make it to a numpy array, so that we can **loop through all the redundant IDs in the *behaviors* dataset and homogenize them** according to our dictionary:

behav = behaviors.copy()
behav = behav.to_numpy()

pbar = ProgressBar()
userIDs_hist_changes = []
userIDs_impr_changes = []
users_to_change = []
articles_to_change_set = set(articles_to_change)

for idx in pbar(range(behav.shape[0])):
    user_row = behav[idx]
    hist_flag = False
    hist = user_row[3]
    hist_list = hist.split()
    hist_set = set(hist_list)
    hist_inter = hist_set & articles_to_change_set
    for art in hist_inter:
        hist_flag = True
        users_to_change.append(idx)
        userIDs_hist_changes.append(user_row[1])
        hist = hist.replace(art, articleID_dict[art])
    if hist_flag:
        behav[idx][3] = hist
    impression_flag = False
    impressions = user_row[4]
    impression_list = [l[:-2] for l in impressions.split()]
    impression_set = set(impression_list)
    for art in (impression_set & articles_to_change_set):
        impression_flag = True
        userIDs_impr_changes.append(user_row[1])
        impressions = impressions.replace(art, articleID_dict[art])
    if impression_flag:        
        behav[idx][4] = impressions

# ## Dropping users who read articles on which there is no information
# 
# We realized, that there are some articles in the history and impression logs without a corresponding entry in the news dataset. So, before saving the processed datasets, we also remove all the users in the behaviors dataset, who read those articles:

unique_articles_behav = set(all_articles)

unique_articles_news = set(news.article_id.unique())

articles_to_drop = unique_articles_behav - unique_articles_news

users_to_drop = []
for i, impr in enumerate(behav):
    hist_set = set(impr[3].split())
    if len(hist_set & articles_to_drop) > 0:
        users_to_drop.append(i)
    
    impr_set = set([art[:-2] for art in impr[4].split()])
    if len(impr_set & articles_to_drop) > 0:
        users_to_drop.append(i)

users_to_drop = list(set(users_to_drop))

behav_new = np.delete(behav, users_to_drop, axis=0)

# Now let's make a **new dataframe out of our processed behavioral data**:

behaviors_new = pd.DataFrame(behav_new, columns=behaviors.columns)

# After the processing from above, the numbers for our *behaviors* dataset now look like this:

print(f'There are now just over {behaviors_new.shape[0]//1000},000 sessions and {len(behaviors_new.user_id.unique())}',
      'individual users in our dataset.')
print(f'The average number of sessions is: {behaviors_new.shape[0] / len(behaviors_new.user_id.unique()):.1f}')

# And also make a new dataframe for the information on **news articles without duplicates**:

news_new = news.drop_duplicates(subset="title", keep='first')

# ### Saving processed datasets
# Now we want to save the processed data and write it to csv files:

behaviors_output_path = dataset_path + "behaviors_processed.csv"
behaviors_new.to_csv(behaviors_output_path, index=False)

news_output_path = dataset_path + "news_processed.csv"
news_new.to_csv(news_output_path, index=False)


# ### Preprocessing for collaborative filtering approaches
# For the deployment of recommender systems which use Collaborative Filtering (CF) techniques, user-article interactions play a pivotal role. Because CF is of great importance to understand modern day recommender systems in general, we too want to construct and discuss different versions of this approach. In order to do this, it is useful to further process our data with repsect to user-article interactions. 
# 
# Because we **only work with the click history** when deploying CF methods, we only need one session per user:
# 

behaviors_cf = behaviors_new.drop_duplicates(subset='user_id').copy()

assert behaviors_cf.shape[0] == behaviors_new.user_id.nunique(),        "User duplicates have not been dropped"

# Now we want to construct a numpy array out of this smaller dataset

behav_cf = behaviors_cf.to_numpy(copy=True)

# so that we can get **two lists with user-article-interactions**. One for training and another one with the last article in history for testing purposes:

uai_train, uai_test = [], []

for row in behav_cf:
    user = row[1]
    hist = row[3].split(' ')
    for art in hist[:-1]:
        uai_train.append([user, art])
    last_art = hist[-1]
    uai_test.append([user, last_art])

#uai_train, uai_test = [], []
#i = 0
#with open(behaviors_output_path, "r") as f:
#    header = f.readline()
#    line = f.readline()
#    print(header)
#    print(line)
#    while line != None and line != "":
#        i += 1
#        print(i, end="\r")
#        row = line.split(",")
#        user = row[1]
#        hist = row[3].split(' ')
#        for art in hist[:-1]:
#            uai_train.append([user, art])
#            
#        last_art = hist[-1]
#        uai_test.append([user, last_art])
#        line = f.readline()


# Now we want to get some **extra user- and article integer IDs**, that we can later use for a **dictionary-of-keys-matrix**, which in turn will be **employed in a neural network**. For our train data, we can do it like this:

uai_train_df = pd.DataFrame(uai_train, columns=['user_id', 'article_id'])

uai_train_df['user_int_id'] = uai_train_df.user_id.astype('category').cat.codes
uai_train_df['article_int_id'] = uai_train_df.article_id.astype('category').cat.codes

# and for our test data, we need to make sure that it contains only the articles which are also in the train data. At first, we need to find those articles:

train_articles = [elem[1] for elem in uai_train]
test_articles = [elem[1] for elem in uai_test]
articles_to_drop = set(test_articles)-set(train_articles)

# With this list, we're now able to reduce our test data:

uai_test_red = [ele for ele in uai_test if ele[1] not in articles_to_drop]

# and can thus create a test dataframe without unknown articles:

uai_test_df = pd.DataFrame(uai_test_red, columns=['user_id', 'article_id'])

# We can then construct two dictionaries, which map the original user and article IDs to the integer IDs:

user_code_dict = pd.Series(uai_train_df.user_int_id.values,
                           index=uai_train_df.user_id).to_dict()

article_code_dict = pd.Series(uai_train_df.article_int_id.values,
                              index=uai_train_df.article_id).to_dict()

uai_test_df['user_int_id'] = [user_code_dict[user] for user in uai_test_df.user_id]
uai_test_df['article_int_id'] = [article_code_dict[art] for art in uai_test_df.article_id]

uai_train_path = dataset_path + f"{dataset_size}_train.csv"
uai_train_df.to_csv(uai_train_path, index=False)

uai_test_path = dataset_path + f"{dataset_size}_test.csv"
uai_test_df.to_csv(uai_test_path, index=False)

# Now we actually **create the dok-matrix**:

num_users, num_articles = uai_train_df.user_id.nunique(), uai_train_df.article_id.nunique()
num_users, num_articles

train_matrix = sp.dok_matrix((num_users, num_articles), dtype=np.float32)

with open(uai_train_path, "r") as f:
    header = f.readline()
    line = f.readline()
    print(header)
    print(line)
    while line != None and line != "":
        line_list = line.split(",")
        user, article = int(line_list[2]), int(line_list[3])
        train_matrix[user, article] = 1.0
        line = f.readline()

if pickle_matrix:
    pickle.dump(train_matrix, open(dataset_path + "train_matrix.pkl", "wb"))

# Later on, when we want to evaluate our recommender systems, we need to **compare the ranking for the known -- but not learned -- interactions with known non-interactions**, so that we can tell how useful our recommendation is: the higher the ranking of the test interaction, the better our model! In order to do this, we want to **extract 99 non read articles for every user in our test set**.

test_interactions = list(zip(uai_test_df.user_int_id, uai_test_df.article_int_id))

num_negatives = 99

negative_interactions = []
for u, i in test_interactions:
    negatives = []
    for t in range(num_negatives):
        j = np.random.randint(num_articles)
        while (u, j) in train_matrix.keys():
            j = np.random.randint(num_articles)
        negatives.append(j)
    negative_interactions.append(negatives)

# **Finally**, we want to **write our one positive interaction along with the randomly generated non interactions into a tsv file** and we're done with the cleaning and preprocessing of the data!

output = negative_interactions[:]

for i in range(len(test_interactions)):
    output[i].insert(0, test_interactions[i])

test_negatives_path = dataset_path + f"{dataset_size}_test_negatives.tsv"
with open(test_negatives_path, 'w') as f:
    for line in output:
        line_str = '\t'.join(str(ele) for ele in line) + "\n"
        f.write(line_str)




