from torchvision import models
from tools import train, draw_loss, test, plot_confusion_matrix, print_statistic_information, draw_accuracy
import pandas as pd
from torch import nn, optim
from sklearn.model_selection import train_test_split
import os

# Set up the dataset path.
dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
path_df_trainset = os.path.join(dir_dataset, 'dataset-master/dataset-master/trainset.csv')
path_df_holdoutset = os.path.join(dir_dataset, 'dataset-master/dataset-master/holdoutset.csv')

# Make the 2-classes dataset.
df_trainset = pd.read_csv(path_df_trainset)
df_trainset_double_1 = df_trainset.loc[df_trainset['label'].isin(['EOSINOPHIL', 'LYMPHOCYTE'])]
df_trainset_double_2 = df_trainset.loc[df_trainset['label'].isin(['NEUTROPHIL', 'MONOCYTE'])]
df_trainset_double_3 = df_trainset.loc[df_trainset['label'].isin(['EOSINOPHIL', 'MONOCYTE'])]
df_trainset_double_1, df_valset_double_1 = train_test_split(df_trainset_double_1, test_size = 0.1)
df_trainset_double_2, df_valset_double_2 = train_test_split(df_trainset_double_2, test_size = 0.1)
df_trainset_double_3, df_valset_double_3 = train_test_split(df_trainset_double_3, test_size = 0.1)
df_holdoutset = pd.read_csv(path_df_holdoutset)
df_holdoutset_double_1 = df_holdoutset.loc[df_holdoutset['label'].isin(['EOSINOPHIL', 'LYMPHOCYTE'])]
df_holdoutset_double_2 = df_holdoutset.loc[df_holdoutset['label'].isin(['NEUTROPHIL', 'MONOCYTE'])]
df_holdoutset_double_3 = df_holdoutset.loc[df_holdoutset['label'].isin(['EOSINOPHIL', 'MONOCYTE'])]

resnet34 = models.resnet34(pretrained=True)
fc_inputs = resnet34.fc.in_features
resnet34.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Linear(10, 4),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Linear(4, 2),
    nn.LogSoftmax(dim=1)
)
resnet34 = resnet34.cuda()

# Re-train the model on the .
epochs = 20
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet34.parameters())
resnet34_binary_1, history = train(resnet34, df_trainset_double_1, df_valset_double_1, epochs, loss_func, optimizer)
resnet34_binary_2, history = train(resnet34, df_trainset_double_2, df_valset_double_2, epochs, loss_func, optimizer)
resnet34_binary_3, history = train(resnet34, df_trainset_double_3, df_valset_double_3, epochs, loss_func, optimizer)

# Using the previous model test on the new dataset.
print('Using the binary models test on the new datasets....')
print("The performance on the ['EOSINOPHIL', 'LYMPHOCYTE'] dataset:")
test(resnet34_binary_1, df_holdoutset_double_1)
print("The performance on the ['NEUTROPHIL', 'MONOCYTE'] dataset:")
test(resnet34_binary_2, df_holdoutset_double_2)
print("The performance on the ['EOSINOPHIL', 'MONOCYTE'] dataset:")
test(resnet34_binary_3, df_holdoutset_double_3)