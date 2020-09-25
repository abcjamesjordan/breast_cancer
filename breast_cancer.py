# Breast Cancer Detection Using Machine Learning

import pandas as pd
import numpy as np
import seaborn as sns
import math

# Importing the data into a pandas DataFrame
bc = pd.read_csv('breast-cancer-wisconsin.csv', header=None, delimiter=',')
bc.set_index(0, inplace=True)
bc.index.name = 'ID'
bc.columns = bc.columns.astype(str)

# Rename columns to letters
col_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
bc.columns = col_replace

# Removing missing value rows
bc = bc[~bc['F'].isin(['?'])]
bc['F'] = pd.to_numeric(bc['F'])

# Split data into train and test sets
bc_len = len(bc.index)
bc_train_len = int(math.floor(bc_len * .6))
bc_test_len = int(math.ceil(bc_len * .2))

bc_train = bc.iloc[0:bc_train_len, :]
bc_dev = bc.iloc[bc_train_len:(bc_train_len + bc_test_len), :]
bc_test = bc.iloc[(bc_train_len + bc_test_len):, :]

bc_train_x = bc_train.iloc[:, :9]
bc_dev_x = bc_dev.iloc[:, :9]
bc_test_x = bc_test.iloc[:, :9]

bc_train_y = bc_train.iloc[:, -1]
bc_dev_y = bc_dev.iloc[:, -1]
bc_test_y = bc_test.iloc[:, -1]

print(bc_train_x.shape, bc_train_y.shape)

#print(bc_train.shape, bc_dev.shape, bc_test.shape)


# Begin the Machine Learning Portion

# Convert data to numpy arrarys

bc_train_x = bc_train_x.to_numpy().T
bc_dev_x = bc_dev_x.to_numpy().T
bc_test_x = bc_test_x.to_numpy().T

bc_train_y = bc_train_y.to_numpy().T
bc_dev_y = bc_dev_y.to_numpy().T
bc_test_y = bc_test_y.to_numpy().T

print(bc_train_x.shape, bc_train_y.shape)













































# End
