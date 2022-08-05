import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

## https://github.com/guillermo-navas-palencia/optbinning
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df2 = pd.read_csv('../data/heloc_dataset_v2.csv')

# 选择要离散化的特征，以及target变量
variable = "mean radius"
variable2 = "AverageMInFile"
# target = 'target'
x = df[variable].values
y = data.target

x1 = df2['AverageMInFile'].values
y1 = df2['target'].values

print(type(x), type(y))
print('================================================')
print(type(x1), type(y1))
for i in range(len(y1)):
    if y1[i] == -1:
        y1[i] = 0



from optbinning import OptimalBinning

optb = OptimalBinning(name=variable2, dtype="numerical", solver="cp")
optb.fit(x1, y1)


print(optb.status)
print(optb.splits)
binning_table = optb.binning_table
print(binning_table)
binning_table.build().to_csv('ttt.csv')
#
binning_table.plot(metric="event_rate")
binning_table.plot(metric="woe")