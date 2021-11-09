from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(arr):
    mean_value = np.mean(arr)
    std_value = np.std(arr)
    arr_nor = arr - mean_value / std_value
    arr_nor = np.array(np.round(arr_nor), dtype=np.uint8)
    return arr_nor

# A function to generate the average picture
def generate_average_picture(paths_img):
    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(paths_img[0]).size
    N = len(paths_img)

    # Create a numpy array of floats to store the average (assume RGB images)
    avg_arr = np.zeros((h, w, 3), np.float64)

    # Build up average pixel intensities, casting each image as an array of floats
    for path_img in paths_img:
        arr_img = np.array(Image.open(path_img), dtype=np.float64)
        avg_arr = avg_arr + arr_img / N

    # Round values in array and cast as 8-bit integer
    avg_arr = np.array(np.round(avg_arr), dtype=np.uint8)

    return avg_arr

dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
dir_xml = os.path.join(dir_row, 'Annotations')
path_df = os.path.join(dir_dataset, 'dataset-master/dataset-master/labels.csv')
print(os.path.exists(dir_img))

# Dorp the empty rows and columns.
df = pd.read_csv(path_df)
df = df.dropna(axis=1, how='all')
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
df = df_modified

df_eos = df.loc[df['Category'] == 'EOSINOPHIL']
df_neu = df.loc[df['Category'] == 'NEUTROPHIL']
df_mon = df.loc[df['Category'] == 'MONOCYTE']
df_lym = df.loc[df['Category'] == 'LYMPHOCYTE']
df_modified = df.loc[df['Category'].isin(['EOSINOPHIL', 'NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE'])]

from sklearn.model_selection import train_test_split
train_set, holdout_set = train_test_split(df_modified, test_size = 0.1)

paths_eos_img = [row['path_img'] for index, row in df_eos.iterrows()]
paths_neu_img = [row['path_img'] for index, row in df_neu.iterrows()]
paths_mon_img = [row['path_img'] for index, row in df_mon.iterrows()]
paths_lym_img = [row['path_img'] for index, row in df_lym.iterrows()]

for index, row in df_modified.iterrows():
    path_img = row['path_img']
    arr_img = np.array(Image.open(path_img), dtype=np.float64)
    arr_nor = normalize(arr_img)
    break
plt.figure("Image")
plt.imshow(arr_nor)