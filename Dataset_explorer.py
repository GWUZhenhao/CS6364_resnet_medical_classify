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

df_neu = pd.DataFrame(columns=['Image', 'Category1', 'Category2'])
df_eos = pd.DataFrame(columns=['Image', 'Category1', 'Category2'])
df_mon = pd.DataFrame(columns=['Image', 'Category1', 'Category2'])
df_bas = pd.DataFrame(columns=['Image', 'Category1', 'Category2'])
df_lym = pd.DataFrame(columns=['Image', 'Category1', 'Category2'])

# generate the dataframe for every categories.
for index, row in df_master_1.iterrows():
    categories = row[2]
    if 'NEUTROPHIL' in categories:
        if ',' in categories:
            category1 = categories.split(',')[0]
            category2 = categories.split(',')[1]
            category2.replace(' ', '')
            df_neu = df_neu.append({'Image': row[1], 'Category1': category1, 'Category2': category2}, ignore_index=True)
        else:
            df_neu = df_neu.append({'Image': row[1], 'Category1': categories, 'Category2': ''}, ignore_index=True)
    if 'EOSINOPHIL' in categories:
        if ',' in categories:
            category1 = categories.split(',')[0]
            category2 = categories.split(',')[1]
            category2.replace(' ', '')
            df_eos = df_eos.append({'Image': row[1], 'Category1': category1, 'Category2': category2}, ignore_index=True)
        else:
            df_eos = df_eos.append({'Image': row[1], 'Category1': categories, 'Category2': ''}, ignore_index=True)
    if 'MONOCYTE' in categories:
        if ',' in categories:
            category1 = categories.split(',')[0]
            category2 = categories.split(',')[1]
            category2.replace(' ', '')
            df_mon = df_mon.append({'Image': row[1], 'Category1': category1, 'Category2': category2}, ignore_index=True)
        else:
            df_mon = df_mon.append({'Image': row[1], 'Category1': categories, 'Category2': ''}, ignore_index=True)
    if 'BASOPHIL' in categories:
        if ',' in categories:
            category1 = categories.split(',')[0]
            category2 = categories.split(',')[1]
            category2.replace(' ', '')
            df_bas = df_bas.append({'Image': row[1], 'Category1': category1, 'Category2': category2},
                                   ignore_index=True)
        else:
            df_bas = df_bas.append({'Image': row[1], 'Category1': categories, 'Category2': ''}, ignore_index=True)
    if 'LYMPHOCYTE' in categories:
        if ',' in categories:
            category1 = categories.split(',')[0]
            category2 = categories.split(',')[1]
            category2.replace(' ', '')
            df_lym = df_lym.append({'Image': row[1], 'Category1': category1, 'Category2': category2},
                                   ignore_index=True)
        else:
            df_lym = df_lym.append({'Image': row[1], 'Category1': categories, 'Category2': ''}, ignore_index=True)

print('The number of Eosinophil is {}'.format(df_eos.shape[0]))
print('The number of Neutrophil is {}'.format(df_neu.shape[0]))
print('The number of Basophil is {}'.format(df_bas.shape[0]))
print('The number of Monocyte is {}'.format(df_mon.shape[0]))
print('The number of Lymphocyte is {}'.format(df_lym.shape[0]))