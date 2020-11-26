import math
import numpy as np
import heapq

def eval_one_rating(idx, model, test_positives, test_negatives, K=10):
    rating = test_positives[idx]
    items = test_negatives[idx]
    u = rating[0]
    get_item = rating[1]
    items.append(get_item)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = model.predict([users, np.array(items)], 
                                batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    
    if get_item in ranklist:
        hr = 1
        i = ranklist.index(get_item)
        ndcg = math.log(2) / math.log(i+2)
        rr = 1/(i+1)
    else:
        hr = 0
        ndcg = 0
        rr = 0
   
    return (hr, ndcg, rr)