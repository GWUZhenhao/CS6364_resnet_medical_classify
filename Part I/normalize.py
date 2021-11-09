from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(arr):
    mean_value = np.mean(arr)
    std_value = np.std(arr)
    arr_nor = (arr - mean_value) / std_value
    return arr_nor

dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
dir_xml = os.path.join(dir_row, 'Annotations')
path_df_modified = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Jupyter_code/modified.csv'

df = pd.read_csv(path_df_modified)
path_img = df.iloc[1]['path_img']
img = Image.open(path_img)

arr = np.array(img, dtype=np.float64)

arr_nor = normalize(arr)

print()