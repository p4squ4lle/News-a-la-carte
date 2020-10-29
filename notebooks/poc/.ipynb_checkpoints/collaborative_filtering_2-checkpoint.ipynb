{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Matrix Factorization for a small subset\n",
    "In this notebook, we're going to build our first recommender system, which follows a **collaborative filtering approach** and only takes into account all the readers and all the articles in a small subset of our data. The goal with this **matrix factorization technique** is to 'learn' two embedding matrices with the respective size of the numbers of readers/articles and an arbitrarily chosen (and thus tunable) size of latent factors. \n",
    "\n",
    "Thus, if we had 10 readers, 5 articles and were to assume we needed 3 latent factors (which could represent implicit, but substantive differences in our reader/article-base), our method will calculate two matrices (a 10 by 3 for the readers and a 3 by 5 for the articles) whose scalar products yield a new matrix the size of our original one (10 x 5), which *approximates* the original matrix best. This optimization problem is typically solved by stochastic gradient descent (although there are, of course, other possibilities) and from a once extremely sparse matrix (obviously, ervery single reader only reads/clicks a tiny fraction of the articles available to us), we get a densely populated table which now contains information on wether some reader might be more or less inclined to read certain articles. \n",
    "\n",
    "The approach might sound a bit dry and mathematic at first, but with the embeddings we actually learn some lower dimensional representations of our readers/articles and can hereby determine *resemblances in preferences*. If you ever wondered how amazon or google knew what you were interested in before you even searched for it: here you go!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "behaviors = pd.read_csv('../../data/mind_small_train/behaviors.tsv', sep=\"\\t\", header=None)\n",
    "news= pd.read_csv('../../data/mind_small_train/news.tsv', sep=\"\\t\", header = None)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The news dataset stores the information of all the news articles (id, header, abstract, ...). It looks like this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>N55528</td>\n",
       "      <td>lifestyle</td>\n",
       "      <td>lifestyleroyals</td>\n",
       "      <td>The Brands Queen Elizabeth, Prince Charles, an...</td>\n",
       "      <td>Shop the notebooks, jackets, and more that the...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAGH0ET.html</td>\n",
       "      <td>[{\"Label\": \"Prince Philip, Duke of Edinburgh\",...</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>N19639</td>\n",
       "      <td>health</td>\n",
       "      <td>weightloss</td>\n",
       "      <td>50 Worst Habits For Belly Fat</td>\n",
       "      <td>These seemingly harmless habits are holding yo...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAB19MK.html</td>\n",
       "      <td>[{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...</td>\n",
       "      <td>[{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>N61837</td>\n",
       "      <td>news</td>\n",
       "      <td>newsworld</td>\n",
       "      <td>The Cost of Trump's Aid Freeze in the Trenches...</td>\n",
       "      <td>Lt. Ivan Molchanets peeked over a parapet of s...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAJgNsz.html</td>\n",
       "      <td>[]</td>\n",
       "      <td>[{\"Label\": \"Ukraine\", \"Type\": \"G\", \"WikidataId...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>N53526</td>\n",
       "      <td>health</td>\n",
       "      <td>voices</td>\n",
       "      <td>I Was An NBA Wife. Here's How It Affected My M...</td>\n",
       "      <td>I felt like I was a fraud, and being an NBA wi...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AACk2N6.html</td>\n",
       "      <td>[]</td>\n",
       "      <td>[{\"Label\": \"National Basketball Association\", ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>N38324</td>\n",
       "      <td>health</td>\n",
       "      <td>medical</td>\n",
       "      <td>How to Get Rid of Skin Tags, According to a De...</td>\n",
       "      <td>They seem harmless, but there's a very good re...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAAKEkt.html</td>\n",
       "      <td>[{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...</td>\n",
       "      <td>[{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        0          1                2  \\\n",
       "0  N55528  lifestyle  lifestyleroyals   \n",
       "1  N19639     health       weightloss   \n",
       "2  N61837       news        newsworld   \n",
       "3  N53526     health           voices   \n",
       "4  N38324     health          medical   \n",
       "\n",
       "                                                   3  \\\n",
       "0  The Brands Queen Elizabeth, Prince Charles, an...   \n",
       "1                      50 Worst Habits For Belly Fat   \n",
       "2  The Cost of Trump's Aid Freeze in the Trenches...   \n",
       "3  I Was An NBA Wife. Here's How It Affected My M...   \n",
       "4  How to Get Rid of Skin Tags, According to a De...   \n",
       "\n",
       "                                                   4  \\\n",
       "0  Shop the notebooks, jackets, and more that the...   \n",
       "1  These seemingly harmless habits are holding yo...   \n",
       "2  Lt. Ivan Molchanets peeked over a parapet of s...   \n",
       "3  I felt like I was a fraud, and being an NBA wi...   \n",
       "4  They seem harmless, but there's a very good re...   \n",
       "\n",
       "                                               5  \\\n",
       "0  https://assets.msn.com/labs/mind/AAGH0ET.html   \n",
       "1  https://assets.msn.com/labs/mind/AAB19MK.html   \n",
       "2  https://assets.msn.com/labs/mind/AAJgNsz.html   \n",
       "3  https://assets.msn.com/labs/mind/AACk2N6.html   \n",
       "4  https://assets.msn.com/labs/mind/AAAKEkt.html   \n",
       "\n",
       "                                                   6  \\\n",
       "0  [{\"Label\": \"Prince Philip, Duke of Edinburgh\",...   \n",
       "1  [{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...   \n",
       "2                                                 []   \n",
       "3                                                 []   \n",
       "4  [{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...   \n",
       "\n",
       "                                                   7  \n",
       "0                                                 []  \n",
       "1  [{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...  \n",
       "2  [{\"Label\": \"Ukraine\", \"Type\": \"G\", \"WikidataId...  \n",
       "3  [{\"Label\": \"National Basketball Association\", ...  \n",
       "4  [{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "news.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "At first, we will only need to work with the behaviors dataset, which looks like this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>U13740</td>\n",
       "      <td>11/11/2019 9:05:58 AM</td>\n",
       "      <td>N55189 N42782 N34694 N45794 N18445 N63302 N104...</td>\n",
       "      <td>N55689-1 N35729-0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>U91836</td>\n",
       "      <td>11/12/2019 6:11:30 PM</td>\n",
       "      <td>N31739 N6072 N63045 N23979 N35656 N43353 N8129...</td>\n",
       "      <td>N20678-0 N39317-0 N58114-0 N20495-0 N42977-0 N...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>U73700</td>\n",
       "      <td>11/14/2019 7:01:48 AM</td>\n",
       "      <td>N10732 N25792 N7563 N21087 N41087 N5445 N60384...</td>\n",
       "      <td>N50014-0 N23877-0 N35389-0 N49712-0 N16844-0 N...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>U34670</td>\n",
       "      <td>11/11/2019 5:28:05 AM</td>\n",
       "      <td>N45729 N2203 N871 N53880 N41375 N43142 N33013 ...</td>\n",
       "      <td>N35729-0 N33632-0 N49685-1 N27581-0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>U8125</td>\n",
       "      <td>11/12/2019 4:11:21 PM</td>\n",
       "      <td>N10078 N56514 N14904 N33740</td>\n",
       "      <td>N39985-0 N36050-0 N16096-0 N8400-1 N22407-0 N6...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   0       1                      2  \\\n",
       "0  1  U13740  11/11/2019 9:05:58 AM   \n",
       "1  2  U91836  11/12/2019 6:11:30 PM   \n",
       "2  3  U73700  11/14/2019 7:01:48 AM   \n",
       "3  4  U34670  11/11/2019 5:28:05 AM   \n",
       "4  5   U8125  11/12/2019 4:11:21 PM   \n",
       "\n",
       "                                                   3  \\\n",
       "0  N55189 N42782 N34694 N45794 N18445 N63302 N104...   \n",
       "1  N31739 N6072 N63045 N23979 N35656 N43353 N8129...   \n",
       "2  N10732 N25792 N7563 N21087 N41087 N5445 N60384...   \n",
       "3  N45729 N2203 N871 N53880 N41375 N43142 N33013 ...   \n",
       "4                        N10078 N56514 N14904 N33740   \n",
       "\n",
       "                                                   4  \n",
       "0                                  N55689-1 N35729-0  \n",
       "1  N20678-0 N39317-0 N58114-0 N20495-0 N42977-0 N...  \n",
       "2  N50014-0 N23877-0 N35389-0 N49712-0 N16844-0 N...  \n",
       "3                N35729-0 N33632-0 N49685-1 N27581-0  \n",
       "4  N39985-0 N36050-0 N16096-0 N8400-1 N22407-0 N6...  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behaviors.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "and needs some column-relabelling:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "behaviors= behaviors.rename(columns={3:'history'})\n",
    "behaviors = behaviors.rename(columns={0:'impression_id'})\n",
    "behaviors = behaviors.rename(columns= {1 : 'user_id'})\n",
    "behaviors = behaviors.rename(columns= {2 : 'time'})\n",
    "behaviors = behaviors.rename(columns= {4 : 'labels'})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we want to check if there are readers with multiple sessions:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "U32146    62\n",
       "U15740    44\n",
       "U20833    41\n",
       "U51286    40\n",
       "U44201    40\n",
       "          ..\n",
       "U8032      1\n",
       "U79126     1\n",
       "U80784     1\n",
       "U43115     1\n",
       "U6836      1\n",
       "Name: user_id, Length: 50000, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behaviors.user_id.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(50000, 156965)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(behaviors.user_id.unique()), len(behaviors.user_id)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Apparently, there are! For matrix factorization, we only want to work with the click history, so let's check whether the click histories for the duplicate users are the same:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "duplicate_users = behaviors.user_id.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "duplicate_users = duplicate_users[duplicate_users!=1].index.to_list()    # create list with the IDs of duplicate users"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ATTENTION: This cell needs some time to compute (~5min), so only uncomment if you have some spare time.\n",
    "# Check whether the click histories of the duplicate users are the same. If not, save the user ID to diff_hist.\n",
    "\n",
    "# diff_hist = []\n",
    "# for user in duplicate_users:\n",
    "#     l = behaviors[behaviors.user_id==user].history.to_list()\n",
    "#     if len(set(l)) != 1:\n",
    "#         diff_hist.append(user)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# len(diff_hist)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "All users with multiple sessions have equal history logs. In contrast, the recommendations and clicks are not the same"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>impression_id</th>\n",
       "      <th>user_id</th>\n",
       "      <th>time</th>\n",
       "      <th>history</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>615</th>\n",
       "      <td>616</td>\n",
       "      <td>U32594</td>\n",
       "      <td>11/10/2019 4:38:09 AM</td>\n",
       "      <td>N54359 N54359 N5227 N16695 N63188 N6253 N60844...</td>\n",
       "      <td>N54595-0 N23757-0 N23820-0 N18572-0 N41220-0 N...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2202</th>\n",
       "      <td>2203</td>\n",
       "      <td>U32594</td>\n",
       "      <td>11/14/2019 2:27:10 AM</td>\n",
       "      <td>N54359 N54359 N5227 N16695 N63188 N6253 N60844...</td>\n",
       "      <td>N41612-0 N16148-0 N3031-0 N51954-0 N2021-0 N33...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4511</th>\n",
       "      <td>4512</td>\n",
       "      <td>U32594</td>\n",
       "      <td>11/14/2019 3:47:55 AM</td>\n",
       "      <td>N54359 N54359 N5227 N16695 N63188 N6253 N60844...</td>\n",
       "      <td>N16419-0 N3167-0 N30071-0 N47721-0 N16148-0 N8...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5095</th>\n",
       "      <td>5096</td>\n",
       "      <td>U32594</td>\n",
       "      <td>11/9/2019 12:36:17 PM</td>\n",
       "      <td>N54359 N54359 N5227 N16695 N63188 N6253 N60844...</td>\n",
       "      <td>N58051-0 N56396-0 N31372-0 N24272-0 N59852-0 N...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5747</th>\n",
       "      <td>5748</td>\n",
       "      <td>U32594</td>\n",
       "      <td>11/12/2019 3:05:21 AM</td>\n",
       "      <td>N54359 N54359 N5227 N16695 N63188 N6253 N60844...</td>\n",
       "      <td>N31978-0 N49157-0 N21741-0 N50675-0 N14184-0 N...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      impression_id user_id                   time  \\\n",
       "615             616  U32594  11/10/2019 4:38:09 AM   \n",
       "2202           2203  U32594  11/14/2019 2:27:10 AM   \n",
       "4511           4512  U32594  11/14/2019 3:47:55 AM   \n",
       "5095           5096  U32594  11/9/2019 12:36:17 PM   \n",
       "5747           5748  U32594  11/12/2019 3:05:21 AM   \n",
       "\n",
       "                                                history  \\\n",
       "615   N54359 N54359 N5227 N16695 N63188 N6253 N60844...   \n",
       "2202  N54359 N54359 N5227 N16695 N63188 N6253 N60844...   \n",
       "4511  N54359 N54359 N5227 N16695 N63188 N6253 N60844...   \n",
       "5095  N54359 N54359 N5227 N16695 N63188 N6253 N60844...   \n",
       "5747  N54359 N54359 N5227 N16695 N63188 N6253 N60844...   \n",
       "\n",
       "                                                 labels  \n",
       "615   N54595-0 N23757-0 N23820-0 N18572-0 N41220-0 N...  \n",
       "2202  N41612-0 N16148-0 N3031-0 N51954-0 N2021-0 N33...  \n",
       "4511  N16419-0 N3167-0 N30071-0 N47721-0 N16148-0 N8...  \n",
       "5095  N58051-0 N56396-0 N31372-0 N24272-0 N59852-0 N...  \n",
       "5747  N31978-0 N49157-0 N21741-0 N50675-0 N14184-0 N...  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behaviors[behaviors.user_id == 'U32594'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(278, 275)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = behaviors[behaviors.user_id == 'U67455'].history.iloc[1].split(' ')\n",
    "len(x), len(set(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It also looks like there are readers who clicked the same articles multiple times. We treat these instances as redundancies here, which -- together with the repeating histories in general -- don't pose a problem for constructing our **original reader-article-matrix**, what we will do in the following:\n",
    "\n",
    "In order to reduce computing time, we want to reduce our dataset to the first 10,000 impressions for this task:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "behav_part_1 = behaviors.iloc[:10000, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9796, 5)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behav_part_1 = behav_part_1.dropna()\n",
    "behav_part_1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>impression_id</th>\n",
       "      <th>user_id</th>\n",
       "      <th>time</th>\n",
       "      <th>history</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>U13740</td>\n",
       "      <td>11/11/2019 9:05:58 AM</td>\n",
       "      <td>N55189 N42782 N34694 N45794 N18445 N63302 N104...</td>\n",
       "      <td>N55689-1 N35729-0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   impression_id user_id                   time  \\\n",
       "0              1  U13740  11/11/2019 9:05:58 AM   \n",
       "\n",
       "                                             history             labels  \n",
       "0  N55189 N42782 N34694 N45794 N18445 N63302 N104...  N55689-1 N35729-0  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behav_part_1.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "id_dict = pd.Series(behav_part_1.user_id.values,index=behav_part_1.impression_id).to_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "behaviors_part_1_set = behav_part_1.set_index('user_id').history.str.split(' ', expand =True).stack().reset_index(1, drop=True).reset_index(name='article')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "user_U67455_set = behaviors_part_1_set[behaviors_part_1_set.user_id == 'U67455']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "N6163     10\n",
       "N59894    10\n",
       "N13231    10\n",
       "N40414     5\n",
       "N6057      5\n",
       "          ..\n",
       "N35494     5\n",
       "N44061     5\n",
       "N23025     5\n",
       "N11597     5\n",
       "N14340     5\n",
       "Name: article, Length: 275, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "user_U67455_set.article.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "behaviors_part_1_set['zus'] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>article</th>\n",
       "      <th>zus</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>U13740</td>\n",
       "      <td>N55189</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U13740</td>\n",
       "      <td>N42782</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>U13740</td>\n",
       "      <td>N34694</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>U13740</td>\n",
       "      <td>N45794</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>U13740</td>\n",
       "      <td>N18445</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>322773</th>\n",
       "      <td>U72585</td>\n",
       "      <td>N14742</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>322774</th>\n",
       "      <td>U72585</td>\n",
       "      <td>N51983</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>322775</th>\n",
       "      <td>U72585</td>\n",
       "      <td>N21189</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>322776</th>\n",
       "      <td>U72585</td>\n",
       "      <td>N46811</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>322777</th>\n",
       "      <td>U72585</td>\n",
       "      <td>N7364</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>322778 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       user_id article  zus\n",
       "0       U13740  N55189    1\n",
       "1       U13740  N42782    1\n",
       "2       U13740  N34694    1\n",
       "3       U13740  N45794    1\n",
       "4       U13740  N18445    1\n",
       "...        ...     ...  ...\n",
       "322773  U72585  N14742    1\n",
       "322774  U72585  N51983    1\n",
       "322775  U72585  N21189    1\n",
       "322776  U72585  N46811    1\n",
       "322777  U72585   N7364    1\n",
       "\n",
       "[322778 rows x 3 columns]"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behaviors_part_1_set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "behaviors_part_1_pivot = behaviors_part_1_set.pivot_table(index='user_id', columns='article', values='zus').fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((8502, 20688), 8502)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behaviors_part_1_pivot.shape, len(behav_part_1.user_id.unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>article</th>\n",
       "      <th>N100</th>\n",
       "      <th>N1000</th>\n",
       "      <th>N10001</th>\n",
       "      <th>N10003</th>\n",
       "      <th>N10009</th>\n",
       "      <th>N1001</th>\n",
       "      <th>N10014</th>\n",
       "      <th>N10016</th>\n",
       "      <th>N10021</th>\n",
       "      <th>N10024</th>\n",
       "      <th>...</th>\n",
       "      <th>N9967</th>\n",
       "      <th>N9969</th>\n",
       "      <th>N997</th>\n",
       "      <th>N9973</th>\n",
       "      <th>N9974</th>\n",
       "      <th>N9977</th>\n",
       "      <th>N9978</th>\n",
       "      <th>N9984</th>\n",
       "      <th>N9992</th>\n",
       "      <th>N9993</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>user_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>U10022</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10043</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10045</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10059</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10062</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 20688 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "article  N100  N1000  N10001  N10003  N10009  N1001  N10014  N10016  N10021  \\\n",
       "user_id                                                                       \n",
       "U10022    0.0    0.0     0.0     0.0     0.0    0.0     0.0     0.0     0.0   \n",
       "U10043    0.0    0.0     0.0     0.0     0.0    0.0     0.0     0.0     0.0   \n",
       "U10045    0.0    0.0     0.0     0.0     0.0    0.0     0.0     0.0     0.0   \n",
       "U10059    0.0    0.0     0.0     0.0     0.0    0.0     0.0     0.0     0.0   \n",
       "U10062    0.0    0.0     0.0     0.0     0.0    0.0     0.0     0.0     0.0   \n",
       "\n",
       "article  N10024  ...  N9967  N9969  N997  N9973  N9974  N9977  N9978  N9984  \\\n",
       "user_id          ...                                                          \n",
       "U10022      0.0  ...    0.0    0.0   0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "U10043      0.0  ...    0.0    0.0   0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "U10045      0.0  ...    0.0    0.0   0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "U10059      0.0  ...    0.0    0.0   0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "U10062      0.0  ...    0.0    0.0   0.0    0.0    0.0    0.0    0.0    0.0   \n",
       "\n",
       "article  N9992  N9993  \n",
       "user_id                \n",
       "U10022     0.0    0.0  \n",
       "U10043     0.0    0.0  \n",
       "U10045     0.0    0.0  \n",
       "U10059     0.0    0.0  \n",
       "U10062     0.0    0.0  \n",
       "\n",
       "[5 rows x 20688 columns]"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "behaviors_part_1_pivot.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy as sp\n",
    "from scipy.sparse.linalg import svds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "b1 = behaviors_part_1_pivot.to_numpy(copy=True)\n",
    "b1_mean = np.mean(b1, axis=1)\n",
    "b1 -= b1_mean.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "U, sigma, Vt = svds(b1, k=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "sigma = np.diag(sigma)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20, 20)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sigma.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "recommendations_df = pd.DataFrame(np.dot(np.dot(U, sigma), Vt) + b1_mean.reshape(-1, 1))\n",
    "recommendations_df.columns = behaviors_part_1_pivot.columns\n",
    "recommendations_df['user_ids'] = behaviors_part_1_pivot.index\n",
    "recommendations_df = recommendations_df.set_index('user_ids')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>article</th>\n",
       "      <th>N100</th>\n",
       "      <th>N1000</th>\n",
       "      <th>N10001</th>\n",
       "      <th>N10003</th>\n",
       "      <th>N10009</th>\n",
       "      <th>N1001</th>\n",
       "      <th>N10014</th>\n",
       "      <th>N10016</th>\n",
       "      <th>N10021</th>\n",
       "      <th>N10024</th>\n",
       "      <th>...</th>\n",
       "      <th>N9967</th>\n",
       "      <th>N9969</th>\n",
       "      <th>N997</th>\n",
       "      <th>N9973</th>\n",
       "      <th>N9974</th>\n",
       "      <th>N9977</th>\n",
       "      <th>N9978</th>\n",
       "      <th>N9984</th>\n",
       "      <th>N9992</th>\n",
       "      <th>N9993</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>user_ids</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>U10022</th>\n",
       "      <td>-0.000280</td>\n",
       "      <td>-0.000243</td>\n",
       "      <td>0.004196</td>\n",
       "      <td>0.000124</td>\n",
       "      <td>-0.000161</td>\n",
       "      <td>-0.000625</td>\n",
       "      <td>0.001704</td>\n",
       "      <td>0.007723</td>\n",
       "      <td>-0.000838</td>\n",
       "      <td>-0.002165</td>\n",
       "      <td>...</td>\n",
       "      <td>0.001413</td>\n",
       "      <td>-0.002237</td>\n",
       "      <td>-0.001649</td>\n",
       "      <td>0.003230</td>\n",
       "      <td>-0.002287</td>\n",
       "      <td>0.004048</td>\n",
       "      <td>0.003238</td>\n",
       "      <td>0.001986</td>\n",
       "      <td>-0.000648</td>\n",
       "      <td>0.000227</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10043</th>\n",
       "      <td>0.000978</td>\n",
       "      <td>0.001051</td>\n",
       "      <td>0.000647</td>\n",
       "      <td>0.001053</td>\n",
       "      <td>0.001231</td>\n",
       "      <td>0.001009</td>\n",
       "      <td>0.001018</td>\n",
       "      <td>0.000300</td>\n",
       "      <td>0.001030</td>\n",
       "      <td>0.001044</td>\n",
       "      <td>...</td>\n",
       "      <td>0.001074</td>\n",
       "      <td>0.000777</td>\n",
       "      <td>0.001134</td>\n",
       "      <td>0.000905</td>\n",
       "      <td>0.001014</td>\n",
       "      <td>0.001096</td>\n",
       "      <td>0.001055</td>\n",
       "      <td>0.001161</td>\n",
       "      <td>0.000896</td>\n",
       "      <td>0.001211</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10045</th>\n",
       "      <td>0.001679</td>\n",
       "      <td>0.001302</td>\n",
       "      <td>0.004273</td>\n",
       "      <td>0.001828</td>\n",
       "      <td>0.001645</td>\n",
       "      <td>0.001876</td>\n",
       "      <td>0.003305</td>\n",
       "      <td>-0.001754</td>\n",
       "      <td>0.001566</td>\n",
       "      <td>0.000592</td>\n",
       "      <td>...</td>\n",
       "      <td>0.003646</td>\n",
       "      <td>-0.001291</td>\n",
       "      <td>-0.000977</td>\n",
       "      <td>0.001018</td>\n",
       "      <td>0.000648</td>\n",
       "      <td>0.002572</td>\n",
       "      <td>0.000337</td>\n",
       "      <td>0.001198</td>\n",
       "      <td>0.001577</td>\n",
       "      <td>0.002121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10059</th>\n",
       "      <td>0.000582</td>\n",
       "      <td>-0.000819</td>\n",
       "      <td>0.001246</td>\n",
       "      <td>-0.000453</td>\n",
       "      <td>-0.001358</td>\n",
       "      <td>-0.000569</td>\n",
       "      <td>-0.000668</td>\n",
       "      <td>0.002088</td>\n",
       "      <td>-0.001098</td>\n",
       "      <td>-0.000746</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000597</td>\n",
       "      <td>0.000823</td>\n",
       "      <td>-0.001739</td>\n",
       "      <td>0.001315</td>\n",
       "      <td>-0.000515</td>\n",
       "      <td>0.000465</td>\n",
       "      <td>0.000518</td>\n",
       "      <td>0.001193</td>\n",
       "      <td>0.001138</td>\n",
       "      <td>0.000827</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U10062</th>\n",
       "      <td>-0.000813</td>\n",
       "      <td>-0.000512</td>\n",
       "      <td>0.001235</td>\n",
       "      <td>-0.002485</td>\n",
       "      <td>-0.002926</td>\n",
       "      <td>-0.003568</td>\n",
       "      <td>0.004105</td>\n",
       "      <td>-0.004009</td>\n",
       "      <td>-0.002954</td>\n",
       "      <td>0.001731</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.001081</td>\n",
       "      <td>0.000073</td>\n",
       "      <td>-0.006820</td>\n",
       "      <td>-0.003728</td>\n",
       "      <td>-0.006758</td>\n",
       "      <td>-0.005362</td>\n",
       "      <td>-0.003010</td>\n",
       "      <td>0.004399</td>\n",
       "      <td>-0.002810</td>\n",
       "      <td>-0.000508</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U9965</th>\n",
       "      <td>-0.000199</td>\n",
       "      <td>-0.000449</td>\n",
       "      <td>-0.000042</td>\n",
       "      <td>-0.000206</td>\n",
       "      <td>-0.000382</td>\n",
       "      <td>0.000117</td>\n",
       "      <td>-0.000284</td>\n",
       "      <td>0.001205</td>\n",
       "      <td>-0.000047</td>\n",
       "      <td>0.001189</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000072</td>\n",
       "      <td>0.000079</td>\n",
       "      <td>0.000357</td>\n",
       "      <td>-0.000794</td>\n",
       "      <td>-0.000149</td>\n",
       "      <td>-0.000279</td>\n",
       "      <td>-0.000371</td>\n",
       "      <td>0.000171</td>\n",
       "      <td>0.000823</td>\n",
       "      <td>0.000022</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U9969</th>\n",
       "      <td>0.000154</td>\n",
       "      <td>0.000264</td>\n",
       "      <td>0.000058</td>\n",
       "      <td>0.000174</td>\n",
       "      <td>0.000133</td>\n",
       "      <td>0.000222</td>\n",
       "      <td>0.000360</td>\n",
       "      <td>0.000657</td>\n",
       "      <td>0.000155</td>\n",
       "      <td>0.000630</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000164</td>\n",
       "      <td>0.000462</td>\n",
       "      <td>0.000299</td>\n",
       "      <td>0.000024</td>\n",
       "      <td>0.000054</td>\n",
       "      <td>-0.000170</td>\n",
       "      <td>0.000124</td>\n",
       "      <td>0.000125</td>\n",
       "      <td>0.000085</td>\n",
       "      <td>0.000277</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U9984</th>\n",
       "      <td>-0.000324</td>\n",
       "      <td>-0.000591</td>\n",
       "      <td>-0.000409</td>\n",
       "      <td>-0.000151</td>\n",
       "      <td>-0.000068</td>\n",
       "      <td>-0.000017</td>\n",
       "      <td>0.001003</td>\n",
       "      <td>0.000484</td>\n",
       "      <td>-0.000443</td>\n",
       "      <td>-0.000383</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000943</td>\n",
       "      <td>-0.000597</td>\n",
       "      <td>-0.001698</td>\n",
       "      <td>-0.000144</td>\n",
       "      <td>0.000664</td>\n",
       "      <td>0.000291</td>\n",
       "      <td>-0.001581</td>\n",
       "      <td>0.001115</td>\n",
       "      <td>0.001792</td>\n",
       "      <td>-0.000385</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U999</th>\n",
       "      <td>0.000278</td>\n",
       "      <td>-0.000227</td>\n",
       "      <td>0.000463</td>\n",
       "      <td>0.000390</td>\n",
       "      <td>0.000119</td>\n",
       "      <td>0.001605</td>\n",
       "      <td>0.000272</td>\n",
       "      <td>-0.003089</td>\n",
       "      <td>0.000424</td>\n",
       "      <td>0.002624</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000443</td>\n",
       "      <td>-0.000559</td>\n",
       "      <td>0.001371</td>\n",
       "      <td>-0.002222</td>\n",
       "      <td>0.001039</td>\n",
       "      <td>-0.000430</td>\n",
       "      <td>0.000136</td>\n",
       "      <td>-0.001938</td>\n",
       "      <td>0.000508</td>\n",
       "      <td>0.000049</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U9991</th>\n",
       "      <td>0.000191</td>\n",
       "      <td>0.000184</td>\n",
       "      <td>0.000197</td>\n",
       "      <td>0.000178</td>\n",
       "      <td>0.000182</td>\n",
       "      <td>0.000176</td>\n",
       "      <td>0.000195</td>\n",
       "      <td>0.000040</td>\n",
       "      <td>0.000181</td>\n",
       "      <td>0.000178</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000182</td>\n",
       "      <td>0.000152</td>\n",
       "      <td>0.000114</td>\n",
       "      <td>0.000158</td>\n",
       "      <td>0.000158</td>\n",
       "      <td>0.000194</td>\n",
       "      <td>0.000126</td>\n",
       "      <td>0.000141</td>\n",
       "      <td>0.000152</td>\n",
       "      <td>0.000186</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8502 rows × 20688 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "article       N100     N1000    N10001    N10003    N10009     N1001  \\\n",
       "user_ids                                                               \n",
       "U10022   -0.000280 -0.000243  0.004196  0.000124 -0.000161 -0.000625   \n",
       "U10043    0.000978  0.001051  0.000647  0.001053  0.001231  0.001009   \n",
       "U10045    0.001679  0.001302  0.004273  0.001828  0.001645  0.001876   \n",
       "U10059    0.000582 -0.000819  0.001246 -0.000453 -0.001358 -0.000569   \n",
       "U10062   -0.000813 -0.000512  0.001235 -0.002485 -0.002926 -0.003568   \n",
       "...            ...       ...       ...       ...       ...       ...   \n",
       "U9965    -0.000199 -0.000449 -0.000042 -0.000206 -0.000382  0.000117   \n",
       "U9969     0.000154  0.000264  0.000058  0.000174  0.000133  0.000222   \n",
       "U9984    -0.000324 -0.000591 -0.000409 -0.000151 -0.000068 -0.000017   \n",
       "U999      0.000278 -0.000227  0.000463  0.000390  0.000119  0.001605   \n",
       "U9991     0.000191  0.000184  0.000197  0.000178  0.000182  0.000176   \n",
       "\n",
       "article     N10014    N10016    N10021    N10024  ...     N9967     N9969  \\\n",
       "user_ids                                          ...                       \n",
       "U10022    0.001704  0.007723 -0.000838 -0.002165  ...  0.001413 -0.002237   \n",
       "U10043    0.001018  0.000300  0.001030  0.001044  ...  0.001074  0.000777   \n",
       "U10045    0.003305 -0.001754  0.001566  0.000592  ...  0.003646 -0.001291   \n",
       "U10059   -0.000668  0.002088 -0.001098 -0.000746  ... -0.000597  0.000823   \n",
       "U10062    0.004105 -0.004009 -0.002954  0.001731  ... -0.001081  0.000073   \n",
       "...            ...       ...       ...       ...  ...       ...       ...   \n",
       "U9965    -0.000284  0.001205 -0.000047  0.001189  ... -0.000072  0.000079   \n",
       "U9969     0.000360  0.000657  0.000155  0.000630  ...  0.000164  0.000462   \n",
       "U9984     0.001003  0.000484 -0.000443 -0.000383  ...  0.000943 -0.000597   \n",
       "U999      0.000272 -0.003089  0.000424  0.002624  ...  0.000443 -0.000559   \n",
       "U9991     0.000195  0.000040  0.000181  0.000178  ...  0.000182  0.000152   \n",
       "\n",
       "article       N997     N9973     N9974     N9977     N9978     N9984  \\\n",
       "user_ids                                                               \n",
       "U10022   -0.001649  0.003230 -0.002287  0.004048  0.003238  0.001986   \n",
       "U10043    0.001134  0.000905  0.001014  0.001096  0.001055  0.001161   \n",
       "U10045   -0.000977  0.001018  0.000648  0.002572  0.000337  0.001198   \n",
       "U10059   -0.001739  0.001315 -0.000515  0.000465  0.000518  0.001193   \n",
       "U10062   -0.006820 -0.003728 -0.006758 -0.005362 -0.003010  0.004399   \n",
       "...            ...       ...       ...       ...       ...       ...   \n",
       "U9965     0.000357 -0.000794 -0.000149 -0.000279 -0.000371  0.000171   \n",
       "U9969     0.000299  0.000024  0.000054 -0.000170  0.000124  0.000125   \n",
       "U9984    -0.001698 -0.000144  0.000664  0.000291 -0.001581  0.001115   \n",
       "U999      0.001371 -0.002222  0.001039 -0.000430  0.000136 -0.001938   \n",
       "U9991     0.000114  0.000158  0.000158  0.000194  0.000126  0.000141   \n",
       "\n",
       "article      N9992     N9993  \n",
       "user_ids                      \n",
       "U10022   -0.000648  0.000227  \n",
       "U10043    0.000896  0.001211  \n",
       "U10045    0.001577  0.002121  \n",
       "U10059    0.001138  0.000827  \n",
       "U10062   -0.002810 -0.000508  \n",
       "...            ...       ...  \n",
       "U9965     0.000823  0.000022  \n",
       "U9969     0.000085  0.000277  \n",
       "U9984     0.001792 -0.000385  \n",
       "U999      0.000508  0.000049  \n",
       "U9991     0.000152  0.000186  \n",
       "\n",
       "[8502 rows x 20688 columns]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recommendations_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>N55528</td>\n",
       "      <td>lifestyle</td>\n",
       "      <td>lifestyleroyals</td>\n",
       "      <td>The Brands Queen Elizabeth, Prince Charles, an...</td>\n",
       "      <td>Shop the notebooks, jackets, and more that the...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAGH0ET.html</td>\n",
       "      <td>[{\"Label\": \"Prince Philip, Duke of Edinburgh\",...</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>N19639</td>\n",
       "      <td>health</td>\n",
       "      <td>weightloss</td>\n",
       "      <td>50 Worst Habits For Belly Fat</td>\n",
       "      <td>These seemingly harmless habits are holding yo...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAB19MK.html</td>\n",
       "      <td>[{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...</td>\n",
       "      <td>[{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>N61837</td>\n",
       "      <td>news</td>\n",
       "      <td>newsworld</td>\n",
       "      <td>The Cost of Trump's Aid Freeze in the Trenches...</td>\n",
       "      <td>Lt. Ivan Molchanets peeked over a parapet of s...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAJgNsz.html</td>\n",
       "      <td>[]</td>\n",
       "      <td>[{\"Label\": \"Ukraine\", \"Type\": \"G\", \"WikidataId...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>N53526</td>\n",
       "      <td>health</td>\n",
       "      <td>voices</td>\n",
       "      <td>I Was An NBA Wife. Here's How It Affected My M...</td>\n",
       "      <td>I felt like I was a fraud, and being an NBA wi...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AACk2N6.html</td>\n",
       "      <td>[]</td>\n",
       "      <td>[{\"Label\": \"National Basketball Association\", ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>N38324</td>\n",
       "      <td>health</td>\n",
       "      <td>medical</td>\n",
       "      <td>How to Get Rid of Skin Tags, According to a De...</td>\n",
       "      <td>They seem harmless, but there's a very good re...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAAKEkt.html</td>\n",
       "      <td>[{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...</td>\n",
       "      <td>[{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        0          1                2  \\\n",
       "0  N55528  lifestyle  lifestyleroyals   \n",
       "1  N19639     health       weightloss   \n",
       "2  N61837       news        newsworld   \n",
       "3  N53526     health           voices   \n",
       "4  N38324     health          medical   \n",
       "\n",
       "                                                   3  \\\n",
       "0  The Brands Queen Elizabeth, Prince Charles, an...   \n",
       "1                      50 Worst Habits For Belly Fat   \n",
       "2  The Cost of Trump's Aid Freeze in the Trenches...   \n",
       "3  I Was An NBA Wife. Here's How It Affected My M...   \n",
       "4  How to Get Rid of Skin Tags, According to a De...   \n",
       "\n",
       "                                                   4  \\\n",
       "0  Shop the notebooks, jackets, and more that the...   \n",
       "1  These seemingly harmless habits are holding yo...   \n",
       "2  Lt. Ivan Molchanets peeked over a parapet of s...   \n",
       "3  I felt like I was a fraud, and being an NBA wi...   \n",
       "4  They seem harmless, but there's a very good re...   \n",
       "\n",
       "                                               5  \\\n",
       "0  https://assets.msn.com/labs/mind/AAGH0ET.html   \n",
       "1  https://assets.msn.com/labs/mind/AAB19MK.html   \n",
       "2  https://assets.msn.com/labs/mind/AAJgNsz.html   \n",
       "3  https://assets.msn.com/labs/mind/AACk2N6.html   \n",
       "4  https://assets.msn.com/labs/mind/AAAKEkt.html   \n",
       "\n",
       "                                                   6  \\\n",
       "0  [{\"Label\": \"Prince Philip, Duke of Edinburgh\",...   \n",
       "1  [{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...   \n",
       "2                                                 []   \n",
       "3                                                 []   \n",
       "4  [{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...   \n",
       "\n",
       "                                                   7  \n",
       "0                                                 []  \n",
       "1  [{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...  \n",
       "2  [{\"Label\": \"Ukraine\", \"Type\": \"G\", \"WikidataId...  \n",
       "3  [{\"Label\": \"National Basketball Association\", ...  \n",
       "4  [{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...  "
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "news.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "titles_dict = pd.Series(news[3].values,index=news[0]).to_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "def give_recommendations(user, n = 5):\n",
    "    recos = recommendations_df.T[user].sort_values().tail(n)\n",
    "    return recos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "article\n",
       "N11101    0.306672\n",
       "N6233     0.320884\n",
       "N41375    0.329777\n",
       "N37509    0.354515\n",
       "N14761    0.456252\n",
       "Name: U91836, dtype: float64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "give_recommendations('U91836')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "article\n",
       "N100      0.000032\n",
       "N1000     0.000729\n",
       "N10001   -0.001700\n",
       "N10003   -0.000208\n",
       "N10009    0.000684\n",
       "            ...   \n",
       "N9977     0.000857\n",
       "N9978    -0.002033\n",
       "N9984     0.001415\n",
       "N9992    -0.000673\n",
       "N9993     0.000366\n",
       "Name: U91836, Length: 20688, dtype: float64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recommendations_df.T['U91836']    #[user].sort_values().tail(n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>N55528</td>\n",
       "      <td>lifestyle</td>\n",
       "      <td>lifestyleroyals</td>\n",
       "      <td>The Brands Queen Elizabeth, Prince Charles, an...</td>\n",
       "      <td>Shop the notebooks, jackets, and more that the...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAGH0ET.html</td>\n",
       "      <td>[{\"Label\": \"Prince Philip, Duke of Edinburgh\",...</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>N19639</td>\n",
       "      <td>health</td>\n",
       "      <td>weightloss</td>\n",
       "      <td>50 Worst Habits For Belly Fat</td>\n",
       "      <td>These seemingly harmless habits are holding yo...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAB19MK.html</td>\n",
       "      <td>[{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...</td>\n",
       "      <td>[{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>N61837</td>\n",
       "      <td>news</td>\n",
       "      <td>newsworld</td>\n",
       "      <td>The Cost of Trump's Aid Freeze in the Trenches...</td>\n",
       "      <td>Lt. Ivan Molchanets peeked over a parapet of s...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAJgNsz.html</td>\n",
       "      <td>[]</td>\n",
       "      <td>[{\"Label\": \"Ukraine\", \"Type\": \"G\", \"WikidataId...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>N53526</td>\n",
       "      <td>health</td>\n",
       "      <td>voices</td>\n",
       "      <td>I Was An NBA Wife. Here's How It Affected My M...</td>\n",
       "      <td>I felt like I was a fraud, and being an NBA wi...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AACk2N6.html</td>\n",
       "      <td>[]</td>\n",
       "      <td>[{\"Label\": \"National Basketball Association\", ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>N38324</td>\n",
       "      <td>health</td>\n",
       "      <td>medical</td>\n",
       "      <td>How to Get Rid of Skin Tags, According to a De...</td>\n",
       "      <td>They seem harmless, but there's a very good re...</td>\n",
       "      <td>https://assets.msn.com/labs/mind/AAAKEkt.html</td>\n",
       "      <td>[{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...</td>\n",
       "      <td>[{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        0          1                2  \\\n",
       "0  N55528  lifestyle  lifestyleroyals   \n",
       "1  N19639     health       weightloss   \n",
       "2  N61837       news        newsworld   \n",
       "3  N53526     health           voices   \n",
       "4  N38324     health          medical   \n",
       "\n",
       "                                                   3  \\\n",
       "0  The Brands Queen Elizabeth, Prince Charles, an...   \n",
       "1                      50 Worst Habits For Belly Fat   \n",
       "2  The Cost of Trump's Aid Freeze in the Trenches...   \n",
       "3  I Was An NBA Wife. Here's How It Affected My M...   \n",
       "4  How to Get Rid of Skin Tags, According to a De...   \n",
       "\n",
       "                                                   4  \\\n",
       "0  Shop the notebooks, jackets, and more that the...   \n",
       "1  These seemingly harmless habits are holding yo...   \n",
       "2  Lt. Ivan Molchanets peeked over a parapet of s...   \n",
       "3  I felt like I was a fraud, and being an NBA wi...   \n",
       "4  They seem harmless, but there's a very good re...   \n",
       "\n",
       "                                               5  \\\n",
       "0  https://assets.msn.com/labs/mind/AAGH0ET.html   \n",
       "1  https://assets.msn.com/labs/mind/AAB19MK.html   \n",
       "2  https://assets.msn.com/labs/mind/AAJgNsz.html   \n",
       "3  https://assets.msn.com/labs/mind/AACk2N6.html   \n",
       "4  https://assets.msn.com/labs/mind/AAAKEkt.html   \n",
       "\n",
       "                                                   6  \\\n",
       "0  [{\"Label\": \"Prince Philip, Duke of Edinburgh\",...   \n",
       "1  [{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...   \n",
       "2                                                 []   \n",
       "3                                                 []   \n",
       "4  [{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...   \n",
       "\n",
       "                                                   7  \n",
       "0                                                 []  \n",
       "1  [{\"Label\": \"Adipose tissue\", \"Type\": \"C\", \"Wik...  \n",
       "2  [{\"Label\": \"Ukraine\", \"Type\": \"G\", \"WikidataId...  \n",
       "3  [{\"Label\": \"National Basketball Association\", ...  \n",
       "4  [{\"Label\": \"Skin tag\", \"Type\": \"C\", \"WikidataI...  "
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "news.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "titles_dict = pd.Series(news[3].values,index=news[0]).to_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "article\n",
       "N12349    0.236248\n",
       "N59704    0.239799\n",
       "N27526    0.259654\n",
       "N4607     0.269513\n",
       "N11231    0.276449\n",
       "N11101    0.306672\n",
       "N6233     0.320884\n",
       "N41375    0.329777\n",
       "N37509    0.354515\n",
       "N14761    0.456252\n",
       "Name: U91836, dtype: float64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "give_recommendations('U91836', n=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "xy=give_recommendations('U91836')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "article\n",
       "N11101    0.306672\n",
       "N6233     0.320884\n",
       "N41375    0.329777\n",
       "N37509    0.354515\n",
       "N14761    0.456252\n",
       "Name: U91836, dtype: float64"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['U10022', 'U10043', 'U10045', 'U10059', 'U10062', 'U10064', 'U10079',\n",
       "       'U10099', 'U10101', 'U10123',\n",
       "       ...\n",
       "       'U9881', 'U9920', 'U9923', 'U9929', 'U994', 'U9965', 'U9969', 'U9984',\n",
       "       'U999', 'U9991'],\n",
       "      dtype='object', name='user_ids', length=8502)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recommendations_df.index.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "behaviors_dev = pd.read_csv('../../data/mind_small_dev/behaviors.tsv', sep=\"\\t\", header=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "beh_num = behav_part_1.to_numpy()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'hist_set' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-52-869705d9d58d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0mtri\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0ms\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mbeh_num\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msplit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m' '\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0ms\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'1'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m     \u001b[0munity\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtri\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m&\u001b[0m \u001b[0mhist_set\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0munity\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m         \u001b[0muser_dic\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0munity\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'hist_set' is not defined"
     ]
    }
   ],
   "source": [
    "user_dic = {}\n",
    "for i in range(beh_num.shape[0]):\n",
    "    tri = [s[:-2] for s in beh_num[i][4].split(' ') if s[-1] == '1']\n",
    "    \n",
    "    unity = set(tri) & hist_set\n",
    "    if len(unity) > 0:\n",
    "        user_dic[i] = list(unity)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_dict = {}\n",
    "for i, s in enumerate(behaviors_part_1_pivot.columns):\n",
    "    map_dict[s] = i"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "map_dict['N10284'], user_dic[21]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.dot(np.dot(U[21, :], sigma), Vt[:, 13175])\n",
    "np.dot(np.dot(U[24, :], sigma), Vt[:, 7831])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = []\n",
    "for k, v in user_dic.items():\n",
    "    for n in v:\n",
    "        news_idx = map_dict[n]\n",
    "        pred = np.dot(np.dot(U[k, :], sigma), Vt[:, news_idx])\n",
    "        results.append(pred + b1_mean[k])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results.sort(reverse=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "erg = pd.DataFrame(np.dot(np.dot(U, sigma), Vt) + b1_mean.reshape(-1, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "erg.columns = behaviors_part_1_pivot.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "erg.iloc[24]['N10016']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.mean(erg.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.std(erg.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "user_dic\n",
    "erg.iloc[24]['N47020']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "recos = []\n",
    "for user, article in user_dic.items():\n",
    "    recos.append(erg.iloc[user][article].to_list())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "recos_2 =[]\n",
    "for x in recos:\n",
    "    for y in x:\n",
    "        recos_2.append(y)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "recos_2 = pd.Series(recos_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "recos_2.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.mean(erg.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.std(erg.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "erg.mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import NMF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "beahviors_np = behaviors_part_1_pivot.to_numpy(copy=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "beahviors_np.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = NMF(n_components=10, init='random', random_state=420)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "W = model.fit_transform(beahviors_np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "H = model.components_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "H.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "W.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nmf_matrix = np.dot(W, H)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nfm_matrix_df = pd.DataFrame(nmf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nfm_matrix_df.columns = behaviors_part_1_pivot.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nfm_matrix_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "recos_nfm = []\n",
    "for user, article in user_dic.items():\n",
    "    recos_nfm.append(nfm_matrix_df.iloc[user][article].to_list())\n",
    "    \n",
    "recos_nfm_2 =[]\n",
    "for x in recos_nfm:\n",
    "    for y in x:\n",
    "        recos_nfm_2.append(y)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:nf] *",
   "language": "python",
   "name": "conda-env-nf-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
