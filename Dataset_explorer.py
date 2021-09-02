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

print(types_1 == types_2)
print(len(types_1))
print(len(types_2))
