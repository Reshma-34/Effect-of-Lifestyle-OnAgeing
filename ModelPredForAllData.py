# -*- coding: utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.metrics import roc_curve



def findEncodedColumn(column, dataEnc_column, start):
    valuesAndCorrespondingColumns = []
    for num, enc_column in enumerate(dataEnc_column.toarray().T): 
        val = column[np.where(enc_column == 1)[0][0]]
        valuesAndCorrespondingColumns.append([val, num + start])
    return valuesAndCorrespondingColumns
        
def dump(name, valuesAndCorrespondingColumns):
    print ' : '.join([name + '_' + str(item[0]) + ' --> ' +str(item[1]) for item in valuesAndCorrespondingColumns])

    
DiseaseNum=1

LR = LogisticRegression(C = 2, class_weight ='auto')  
RF = RandomForestClassifier(n_estimators = 300, class_weight = 'auto')
GBM = GradientBoostingClassifier (n_estimators = 300)
    
dataCSV = pd.read_csv('ImputedAllDataRemovedCorr.csv', sep=',', error_bad_lines=False, index_col=False, low_memory=False)
data = dataCSV.as_matrix()
header = dataCSV.columns.values

CategoricalSelFeatures = [93, 58, 0, 113, 61, 42, 92, 12, 54, 81, 99, 26, 102, 44, 84, 110, 35, 27, 29, 25, 77, 118, 21, 106, 88, 123, 125, 98] ## Arthritis
#CategoricalSelFeatures = [79, 86, 80, 89, 0, 94, 53, 81, 114, 99, 76, 49, 44, 34, 30, 61, 84, 77, 122, 103, 104] ## Angina
#CategoricalSelFeatures = [61, 84, 58, 12, 89, 82, 8, 7, 1, 110, 49, 31, 6, 30, 83, 112, 95, 71, 24, 91, 102, 79, 11, 2, 53, 85, 59, 73, 64, 125, 90, 109] ## Chronic_Lung

CategoricalSelFeatures = [4 + i for i in CategoricalSelFeatures]
ContinuousFeature=['V21', 'V55', 'V121', 'V117']
ContiHeader=[]
for conti in ContinuousFeature:
    ContiHeader.append(np.where(header==conti)[0][0])
    
header_categorical = header[CategoricalSelFeatures]

X= data[:,CategoricalSelFeatures]
enc = OneHotEncoder()
Xenc = enc.fit_transform(X)

enc_features_indices = [0]    
for i in range(len(enc.feature_indices_) - 1):
    enc_features_indices.append((enc.active_features_ < enc.feature_indices_[i+1]).sum())

encoded_column_names = []    
for num, (name, column) in enumerate(zip(header_categorical, X.T)):
    enc_features_columns = range(enc_features_indices[num],enc_features_indices[num+1])
    dataEnc_column = Xenc[:, enc_features_columns]
    valuesAndCorrespondingColumns = findEncodedColumn(column, dataEnc_column, enc_features_indices[num])
    encoded_column_names.extend([name + '_' + str(item[0]) for item in valuesAndCorrespondingColumns])
    
encoded_column_names.extend(ContinuousFeature)

encoded_column_names = np.array(encoded_column_names)

num_train = 10477
train_data=hstack((Xenc[:num_train], data[:num_train,ContiHeader]))
test_data=hstack((Xenc[num_train:], data[num_train:,ContiHeader]))

Xtrain, Xtest = train_data, test_data

Ytrain = data[:num_train, DiseaseNum ]

Cvals = np.logspace(-1, 0.5, 10)
bestscore = [0.1, 0]
for C in Cvals:
    LR.C = C
    score = np.mean(cross_val_score(LR, Xtrain, Ytrain, scoring = 'roc_auc', cv =10))
    if score > bestscore[1]:
        bestscore = [C, score]
        
LR.C = bestscore[0]

score_LR = np.mean(cross_val_score(LR, Xtrain, Ytrain, scoring = 'roc_auc', cv =10))
print "LR score:", score_LR

score_RF = np.mean(cross_val_score(RF, Xtrain, Ytrain, scoring = 'roc_auc', cv =10))
print "RF score:", score_RF
score_GBM = np.mean(cross_val_score(GBM, Xtrain.toarray(), Ytrain, scoring = 'roc_auc', cv =10))
print "GBM score:", score_GBM

LR.fit(Xtrain, Ytrain)
#~ RF.fit(Xtrain, Ytrain)
#~ GBM.fit(Xtrain.toarray(), Ytrain)

Ytest = LR.predict(Xtest)
PredY = np.vstack((data[num_train:, 0], Ytest)).T

coeff = np.vstack((encoded_column_names, LR.coef_)).T
np.savetxt("Coefficient_LR_AR.csv", coeff, delimiter=',', fmt ='%s')


fpr_LR, tpr_LR, thresholds = roc_curve(Ytrain, LR.predict_proba(Xtrain)[:,1])
plt.figure(figsize = (5,4.5))
plt.plot(fpr_LR, tpr_LR, '-s', color = 'red', mec = 'red', mfc = 'none', ms = 1)

#~ fpr_RF, tpr_RF, thresholds = roc_curve(Ytrain, RF.predict_proba(Xtrain)[:,1])
#~ plt.plot(fpr_RF, tpr_RF, '-o', color = 'green', mec = 'green', mfc = 'none', ms = 4)
#~ 
#~ fpr_GBM, tpr_GBM, thresholds = roc_curve(Ytrain, GBM.predict_proba(Xtrain.toarray())[:,1])
#~ plt.plot(fpr_GBM, tpr_GBM, '-^', color = 'blue', mec = 'blue', mfc = 'none', ms = 4)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.subplots_adjust(bottom = 0.13, left = 0.13)
#~ plt.legend(('LR','RF','GBM'), frameon = False)

plt.savefig("ROC_ARModel.png", dpi = 300)
plt.show()

np.savetxt("Prediction\CAX_Arthritis_Pred_09102015.csv", PredY, delimiter=',', fmt ='%s')
#np.savetxt("Prediction\CAX_Angina_Pred_09102015.csv", PredY, delimiter=',', fmt ='%s')
#np.savetxt("Prediction\CAX_ChronicLung_Pred_09102015.csv", PredY, delimiter=',', fmt ='%s')
