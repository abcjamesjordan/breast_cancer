# Breast Cancer Detection Using Machine Learning

import pandas as pd
import numpy as np
import seaborn as sns

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

print(bc.head())

bc.to_csv('bc_cleaned.csv')
