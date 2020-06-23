import numpy as np
import pandas as pd

def findNumPairs(arr1, arr2, val1, val2):
	return np.where(np.logical_and(arr1 == val1, arr2 == val2))[0].shape[0]

def findBestPair(arr1, arr2, vals1, vals2):
	bestPair = [vals1[0], vals2[0]]
	maxNumPairs = 0
	for v1 in vals1:
		for v2 in vals2:
			numPairs = findNumPairs(arr1, arr2, v1, v2)
			if numPairs > maxNumPairs:
				bestPair = [v1, v2]
				maxNumPairs = numPairs
			
	return bestPair, maxNumPairs

	
	
def findCorrelation(arr1, arr2):
	vals1 = np.unique(arr1)
	vals2 = np.unique(arr2)
	num1 = vals1.shape[0]
	num2 = vals2.shape[0]
	totalPairs = 0
	allPairs = []
	for counter in range(min(num1, num2)):
		bestPair, maxNumPairs = findBestPair(arr1, arr2, vals1, vals2)
		totalPairs += maxNumPairs
		allPairs.append(bestPair)
		vals1 = np.delete(vals1, np.where(vals1 == bestPair[0]))
		vals2 = np.delete(vals2, np.where(vals2 == bestPair[1]))
		
	return totalPairs*1.0/arr1.shape[0], allPairs
	

dataCSV = pd.read_csv('ImputedCategoricalOrdinal_train.csv', sep=',', error_bad_lines=False, index_col=False, low_memory=False)
data = dataCSV.as_matrix()
#data= data[:,4:]
header = dataCSV.columns.values


op = open('CorrCategoricalOrdinal.csv',"w")  
op.write(" "+","+','.join(header[4:])+"\n")

for i in range(4, data.shape[1]):
    op.write(header[i]+",")
    for j in range(4, data.shape[1]):
        corr_coeff=findCorrelation(data.T[:,i], data.T[:,j])
        op.write(str(corr_coeff[0])+",")  
    op.write("\n")
op.close()
