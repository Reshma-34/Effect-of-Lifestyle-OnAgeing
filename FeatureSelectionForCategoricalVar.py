from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy as np
import operator

def find_index(act_f, c_range):
    indexEnc = []
    for act_idx, feature in enumerate(act_f):
        if(feature >= c_range[0] and feature < c_range[1]):
            indexEnc.append(act_idx)
    return indexEnc
    
def selectFeatures(X, Y, num_select = 50):
    enc = OneHotEncoder()
    Xenc = enc.fit_transform(X)
    act_feature= enc.active_features_
    feature_idx= enc.feature_indices_
    enc_features= []
    for col in range(feature_idx.shape[0]-1):
        col_range=[feature_idx[col], feature_idx[col+1]]
        indexF= find_index(act_feature, col_range)
        enc_features.append(indexF)
    selected_features = []
    print "Selecting features now ... "
    while len(selected_features) < num_select:
        scores = []
        for i in range(X.shape[1]):
            
            if i not in selected_features:
                test_features = selected_features + [i]
                test_features_enc = [enc_features[i] for i in test_features]
                test_features_enc = [item for sublist in test_features_enc for item in sublist]

                Xenc_test = Xenc[:,test_features_enc]
                score = np.mean(cross_val_score(LR, Xenc_test, Y, scoring = 'roc_auc', cv =10))
                scores.append([i, score])
              
        scores.sort(key = operator.itemgetter(1), reverse = True)

        selected_features.append(scores[0][0])
        this_score = scores[0][1]
        print "currently, " + str(len(selected_features)) + " selected - ", selected_features
        print "current score:", this_score

    return selected_features

LR = LogisticRegression(C = 2)
dataCSV = pd.read_csv('ImputedCategoricalOrdinal_train_RemovedCorr.csv', sep=',', error_bad_lines=False, index_col=False, low_memory=False)
data = dataCSV.as_matrix()
header = dataCSV.columns.values
X= data[:,4:]
print selectFeatures(X, data.T[2])
