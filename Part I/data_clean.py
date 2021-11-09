import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
path_df = os.path.join(dir_dataset, 'dataset-master/dataset-master/labels.csv')

# Dorp the empty rows and columns.
df = pd.read_csv(path_df)
df = df.dropna(axis=1, how='all')
num_null = df.isnull().sum()['Category']
df = df.dropna(axis=0, how='any')

# If the image is not exist, drop the row.
cols = df.columns.to_list()
cols.append('path_img')
df_modified = pd.DataFrame(columns=df.columns)
for index, row in df.iterrows():
    # This 'path_img' will record the addresses of the images
    row['path_img'] = os.path.join(dir_img, 'BloodImage_' + str(row['Image']).zfill(5) + '.jpg')
    if os.path.exists(row['path_img']):
        df_modified = df_modified.append(row, ignore_index=True)
    else:
        num_null += 1
df = df_modified

# We implement dropping double class here.
df_eos = df.loc[df['Category'] == 'EOSINOPHIL']
df_neu = df.loc[df['Category'] == 'NEUTROPHIL']
df_mon = df.loc[df['Category'] == 'MONOCYTE']
df_lym = df.loc[df['Category'] == 'LYMPHOCYTE']

double_class = [i for i in df['Category'].tolist() if ',' in i]
df_double_class = df.loc[df['Category'].isin(double_class)]

df_modified = df.loc[df['Category'].isin(['EOSINOPHIL', 'NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE'])]

train_set, holdout_set = train_test_split(df_modified, test_size = 0.1)
df_holdoutset = holdout_set[['path_img', 'Category']]
df_holdoutset.columns = ['path_image', 'label']

num_eos = df_eos.shape[0]
num_neu = df_neu.shape[0]
num_mon = df_mon.shape[0]
num_lym = df_lym.shape[0]
num_double_class = df_double_class.shape[0]

print('The number of Eosinophil is {}'.format(num_eos))
print('The number of Neutrophil is {}'.format(num_neu))
print('The number of Monocyte is {}'.format(num_mon))
print('The number of Lymphocyte is {}'.format(num_lym))
print('The number of double class is {}'.format(num_double_class))
print('The number of null is {}'.format(num_null))

features = train_set.path_img.to_list()
labels = train_set.Category.to_list()

# randomly dropping some of the Neutrophil data
import random
neu_indexes = [index for index, label in enumerate(labels) if label == 'NEUTROPHIL']
drop_percentage = 20 # Change dropping rate here!
k = len(neu_indexes) * drop_percentage // 100
indicies = random.sample(neu_indexes, k)
new_labels = [labels[i] for i, value in enumerate(labels) if i not in indicies]
new_features = [features[i] for i, value in enumerate(features) if i not in indicies]
print(len(new_labels))
print(len(labels))
print(len(new_features))
print(len(features))

trainset_data = {'path_image':new_features,
                 'label':new_labels}
df_trainset = pd.DataFrame(trainset_data)
df_trainset, df_valset = train_test_split(df_trainset, test_size = 0.1)

# Handle the data imbalance by double some classes
# Double the Eosinophil class, 9 * Monocyte class and 6 * Lymphocyte class
eos_df_train = df_trainset[df_trainset['label'] == 'EOSINOPHIL']
mon_df_train = df_trainset[df_trainset['label'] == 'MONOCYTE']
lym_df_train = df_trainset[df_trainset['label'] == 'LYMPHOCYTE']
new_df_trainset=pd.concat([df_trainset,
                           eos_df_train,
                           mon_df_train, mon_df_train, mon_df_train,
                           lym_df_train, lym_df_train, lym_df_train,lym_df_train],axis=0,ignore_index=True)

new_eos_df_train = new_df_trainset[new_df_trainset['label'] == 'EOSINOPHIL']
new_mon_df_train = new_df_trainset[new_df_trainset['label'] == 'MONOCYTE']
new_lym_df_train = new_df_trainset[new_df_trainset['label'] == 'LYMPHOCYTE']
print('num of Eos from {} to {}'.format(eos_df_train.shape[0], new_eos_df_train.shape[0]))
print('num of Mon from {} to {}'.format(mon_df_train.shape[0], new_mon_df_train.shape[0]))
print('num of Lym from {} to {}'.format(lym_df_train.shape[0], new_lym_df_train.shape[0]))
print('num of Neu is {}'.format(new_df_trainset[new_df_trainset['label'] == 'NEUTROPHIL'].shape[0]))
df_trainset = new_df_trainset

df_holdoutset.to_csv(os.path.join(dir_row, 'holdoutset.csv'), index=None)
df_valset.to_csv(os.path.join(dir_row, 'valset.csv'), index=None)
df_trainset.to_csv(os.path.join(dir_row, 'trainset.csv'), index=None)
