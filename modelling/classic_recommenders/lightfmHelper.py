import numpy as np
from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm.evaluation import  auc_score, reciprocal_rank


def evaluate(model, train, test, hybrid=False, features=None):
    
    if hybrid:
        auc_train = np.mean(auc_score(model, train, item_features=features))
        pre_train = np.mean(precision_at_k(model, train, item_features=features))
        mrr_train = np.mean(reciprocal_rank(model, train, item_features=features))
        
        auc_test = np.mean(auc_score(model, test, item_features=features))
        pre_test = np.mean(precision_at_k(model, test, item_features=features))
        mrr_test = np.mean(reciprocal_rank(model, test, item_features=features))   
    else:
        auc_train = np.mean(auc_score(model, train))
        pre_train = np.mean(precision_at_k(model, train))
        mrr_train = np.mean(reciprocal_rank(model, train))
        
        auc_test = np.mean(auc_score(model, test))
        pre_test = np.mean(precision_at_k(model, test))
        mrr_test = np.mean(reciprocal_rank(model, test))    
    
    res_dict = {'auc_train': auc_train, 
                'pre_train': pre_train,
                'mrr_train': mrr_train, 
                'auc_test': auc_test, 
                'pre_test': pre_test, 
                'mrr_test': mrr_test}
                  
    print('The AUC Score is in training/validation:                 ',
          auc_train,' / ', auc_test)
    print('The mean precision at k Score in training/validation is: ',
          pre_train, ' / ', pre_test)
    print('The mean reciprocal rank in training/validation is:      ', 
          mrr_train, ' / ', mrr_test)
    print('_________________________________________________________')
    
    return res_dict