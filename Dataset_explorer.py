import numpy as np
import pandas as pd
import os

# This viable is the address of the dataset:
dir_dataset = 'C:/Users/36394/Documents/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'

# Read the csv into dataframe
path_df_master_1 = os.path.join(dir_dataset, 'dataset-master/dataset-master/labels.csv')
df_master_1 = pd.read_csv(path_df_master_1, keep_default_na=False)
path_df_master_2 = os.path.join(dir_dataset, 'dataset2-master/dataset2-master/labels.csv')
df_master_2 = pd.read_csv(path_df_master_2, keep_default_na=False)


types_1 = [i for i in df_master_1['Category'] if i != '']
types_2 = [i for i in df_master_2['Category'] if i != '']

# Check 2 sub-dataset is equal or not
print('two dataframe is equal or not: {}'.format(types_1 == types_2))
print('The length of first dataframe is: {}.'.format(len(types_1)))

# Explore the types of the cell
types = [i for i in np.unique(types_1) if ',' not in i]
print('There are {} types of cells'.format(len(types)))
print('The types of dataset is {}'.format(types))
df_Neu = df_master_1[df_master_1['Category'] == 'NEUTROPHIL']
df_EOS = df_master_1[df_master_1['Category'] == 'EOSINOPHIL']
df_Mon = df_master_1[df_master_1['Category'] == 'MONOCYTE']
df_Bas = df_master_1[df_master_1['Category'] == 'BASOPHIL']
df_Lym = df_master_1[df_master_1['Category'] == 'LYMPHOCYTE']
print('The number of Eosinophil is {}'.format(df_EOS))
print('The number of Neutrophil is {}'.format(df_Neu))
print('The number of Basophil is {}'.format(df_Bas))
print('The number of Monocyte is {}'.format(df_Mon))
print('The number of Lymphocyte is {}'.format(df_Lym))