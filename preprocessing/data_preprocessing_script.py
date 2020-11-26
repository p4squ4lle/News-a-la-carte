#!/usr/bin/env python
# coding: utf-8

# Preprocessing and Cleaning of the Data

import pandas as pd
import numpy as np
from progressbar import ProgressBar

import os

dirname = os.path.dirname(__file__)
behaviors_fn = os.path.join(
    dirname, '../../data/mind_small_train/behaviors.tsv'
)
news_fn = os.path.join(
    dirname, '../../data/mind_small_train/news.tsv'
)
# The news dataset stores the information of all the news articles (id, header, abstract, ...).
print("Loading the data sets ...")
behaviors = pd.read_csv(behaviors_fn, sep="\t", header=None)
news = pd.read_csv(news_fn, sep="\t", header = None)


# Renaming news columns
print("Renaming news and behaviors columns ...")
news = news.rename(columns={0:'article_id'})
news = news.rename(columns={1:'category'})
news = news.rename(columns={2:'subcategory'})
news = news.rename(columns={3:'title'})
news = news.rename(columns={4:'abstract'})
news = news.rename(columns={5:'url'})
news = news.rename(columns={6:'title_entities'})
news = news.rename(columns={7:'abstract_entities'})

# Apparently there are news articles with multiple IDs. We don't just want to drop them, because this would result in a loss of useful information concerning the click behaviors and reading histories in the bahaviors dataset.

# Renaming behaviors columns
behaviors = behaviors.rename(columns={3:'history'})
behaviors = behaviors.rename(columns={0:'impression_id'})
behaviors = behaviors.rename(columns= {1 : 'user_id'})
behaviors = behaviors.rename(columns= {2 : 'time'})
behaviors = behaviors.rename(columns= {4 : 'impressions'})

behaviors.dropna(inplace=True)

# With different IDs for the de facto same articles we would not be able to track similarities among users sufficiently. In the following, we will replace every redundant ID with the first ID for the respective article. In order to this we create a subset with all the duplicates in it (duplis_title).
print('Creating duplicate title list and the articleID dictionary...')

duplis_title = news[news.duplicated(subset="title", keep=False)]

title_set = duplis_title['title'].unique()

article_list = []
for title in title_set:
    x = duplis_title[duplis_title['title']==title]['article_id'].to_list()
    article_list.append(x)

article_list[:5]

# With this article_list we have a list which contains article IDs for every article which has multiple IDs in our original dataset. 

# Now we want to make a dictionary called articleID_dict, which maps all the redundant IDs (keys) to a single ID (value):

articleID_dict = {}
articles_to_change = []
for article in article_list:
    value = article[0]
    keys = article [1:]
    for k in keys:
        articleID_dict[k] = value
        articles_to_change.append(k)

# Let's make a copy of the original behaviors dataframe so that we can compare it to the one we are constructing.

behav = behaviors.copy()

# In the cell below we loop over all the reader sessions in the behaviors dataset and replace the redundant article IDs with the ones specified in our article ID dictionary. The loop takes some time, but we will only have to run it once and can save the resulting dataframe to a new CSV file, that we can work with later on.
print("Replacing the redundant article IDs with the ones specified", 
      "in the article ID dictionary.",
      "This will take some time. Please be patient...")

userIDs_hist_changes = []
userIDs_impr_changes = []

pbar = ProgressBar()
for idx in pbar(behav.index):
    user_row = behav.loc[idx, :]
    
    hist_flag = False
    hist = user_row.history
    hist_list = hist.split()
    
    for art in hist_list:
        if art in articles_to_change:
            hist_flag = True
            userIDs_hist_changes.append(user_row["user_id"])
            hist = hist.replace(art, articleID_dict[art])
    if hist_flag:
        behav.loc[idx, "history"] = hist
    
    impression_flag = False
    impressions = user_row.impressions
    impression_list = [l[:-2] for l in impressions.split()]
    
    for l in impression_list:
        if l in articles_to_change:
            impression_flag = True
            userIDs_impr_changes.append(user_row["user_id"])
            impressions = impressions.replace(l, articleID_dict[l])
    if impression_flag:        
        behav.loc[idx, "impressions"] = impressions

        
print("Saving the newly generated dataframes")        
behav.to_csv("../../data/mind_small_train/behaviors_processed_from_script.csv",
             index=False)

news_dropped = news.drop_duplicates(subset="title", keep='first')
news_dropped.to_csv("../../data/mind_small_train/news_processed_from_script.csv", 
                    index=False)

print("Finished!")