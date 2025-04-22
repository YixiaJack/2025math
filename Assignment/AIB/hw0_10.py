import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (a) Load the Boston dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = [
    "CRIM",    
    "ZN",      
    "INDUS",   
    "CHAS",    
    "NOX",     
    "RM",     
    "AGE",    
    "DIS",     
    "RAD",    
    "TAX",    
    "PTRATIO", 
    "B",      
    "LSTAT"    
]
boston = pd.DataFrame(data, columns=feature_names)
boston['MEDV'] = target

# (b) Display basic dataset information
print("Boston dataset shape (rows, columns):", boston.shape)
print("Explanation: Each row represents a Boston suburb (or census tract), and each column is a variable.")

# (c) Create pairwise scatterplots (scatterplot matrix) of the predictors
pd.plotting.scatter_matrix(boston, figsize=(15, 15), diagonal='hist', alpha=0.7)
plt.suptitle("Scatterplot Matrix of Boston Housing Data", fontsize=16)
plt.show()

# (d) Analyze predictors associated with per capita crime rate (CRIM)

corr_matrix = boston.corr()
print("\nCorrelation of predictors with CRIM:")
print(corr_matrix["CRIM"].sort_values(ascending=False))

plt.figure(figsize=(8, 6))
plt.scatter(boston['LSTAT'], boston['CRIM'], edgecolor='k', alpha=0.7)
plt.xlabel("LSTAT (% lower status population)")
plt.ylabel("CRIM (per capita crime rate)")
plt.title("CRIM vs. LSTAT")
plt.show()

# (e) Examine the range of selected predictors: CRIM, TAX, and PTRATIO

print("\nRange for selected predictors:")
print("CRIM - min: {:.2f}, max: {:.2f}".format(boston['CRIM'].min(), boston['CRIM'].max()))
print("TAX  - min: {:.2f}, max: {:.2f}".format(boston['TAX'].min(), boston['TAX'].max()))
print("PTRATIO - min: {:.2f}, max: {:.2f}".format(boston['PTRATIO'].min(), boston['PTRATIO'].max()))

# (f) How many suburbs bound the Charles River? (CHAS = 1)

chas_count = boston['CHAS'].sum()
print("\nNumber of suburbs that bound the Charles River (CHAS = 1):", int(chas_count))

# (g) What is the median pupil-teacher ratio among the towns?
median_ptratio = boston['PTRATIO'].median()
print("\nMedian pupil-teacher ratio (PTRATIO):", median_ptratio)

# (h) Identify the suburb with the lowest median value (MEDV) and compare its characteristics

min_medv_index = boston['MEDV'].idxmin()
lowest_medv_town = boston.loc[min_medv_index]
print("\nSuburb with the lowest median value (MEDV):")
print("Index:", min_medv_index)
print(lowest_medv_town)

print("\nOverall range for each predictor (excluding MEDV):")
for col in boston.columns:
    if col != 'MEDV':
        print(f"{col}: min = {boston[col].min()}, max = {boston[col].max()}")

# (i) Count the number of suburbs with average number of rooms per dwelling > 7 and > 8
num_more_than_7 = (boston['RM'] > 7).sum()
num_more_than_8 = (boston['RM'] > 8).sum()
print("\nNumber of suburbs with more than 7 rooms per dwelling:", num_more_than_7)
print("Number of suburbs with more than 8 rooms per dwelling:", num_more_than_8)

if num_more_than_8 > 0:
    print("\nDetails of suburbs with more than 8 rooms per dwelling:")
    print(boston[boston['RM'] > 8])

# (Answers):

"""
Boston dataset shape (rows, columns): (506, 14)
Explanation: Each row represents a Boston suburb (or census tract), and each column is a variable.

Correlation of predictors with CRIM:
CHAS       0.2203
INDUS     -0.0510
RM        -0.2199
DIS        0.2487
ZN         0.1980
LSTAT     -0.3883
PTRATIO   -0.4093
NOX       -0.4069
TAX       -0.3857
AGE        0.3567
B          0.3333
CRIM       1.0000
RAD       -0.1020
Name: CRIM, dtype: float64

Range for selected predictors:
CRIM - min: 0.01, max: 88.98
TAX  - min: 187.00, max: 711.00
PTRATIO - min: 12.60, max: 22.00

Number of suburbs that bound the Charles River (CHAS = 1): 35

Median pupil-teacher ratio (PTRATIO): 19.05

Suburb with the lowest median value (MEDV):
Index: 343
CRIM         6.5750
ZN          18.00
INDUS       18.10
CHAS         0.00
NOX          0.7130
RM           5.9940
AGE        100.00
DIS          1.5459
RAD          5.00
TAX        311.00
PTRATIO     21.00
B          396.90
LSTAT        5.1900
MEDV         9.00
Name: 343, dtype: float64

Overall range for each predictor (excluding MEDV):
CRIM: min = 0.00632, max = 88.9762
ZN: min = 0.0, max = 100.0
INDUS: min = 0.46, max = 27.74
CHAS: min = 0.0, max = 1.0
NOX: min = 0.385, max = 0.871
RM: min = 3.561, max = 8.78
AGE: min = 2.9, max = 100.0
DIS: min = 1.1296, max = 12.1265
RAD: min = 1.0, max = 24.0
TAX: min = 187.0, max = 711.0
PTRATIO: min = 12.6, max = 22.0
B: min = 0.32, max = 396.9
LSTAT: min = 1.73, max = 37.97

Number of suburbs with more than 7 rooms per dwelling: 64
Number of suburbs with more than 8 rooms per dwelling: 13

Details of suburbs with more than 8 rooms per dwelling:
     CRIM     ZN  INDUS  CHAS    NOX     RM   AGE    DIS  RAD    TAX  \
<displayed DataFrame slice with 13 rows>

   PTRATIO      B  LSTAT  MEDV
<displayed DataFrame slice with 13 rows>
"""