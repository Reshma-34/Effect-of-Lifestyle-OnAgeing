import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr   

dataCSV = pd.read_csv('ImputedContinuous_train.csv', sep=',', error_bad_lines=False, index_col=False, low_memory=False)
data = dataCSV.as_matrix()
header = dataCSV.columns.values

op = open('CorrContinuous.csv',"w")  
op.write(" "+","+','.join(header[4:])+"\n")

for i in range(4,data.shape[1]):
    op.write(header[i]+",")
    for j in range(4, data.shape[1]):
        pear_coeff=pearsonr(data.T[:,i], data.T[:,j])
        op.write(str(pear_coeff[0])+",")  
    op.write("\n")
op.close()