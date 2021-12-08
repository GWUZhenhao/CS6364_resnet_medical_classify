import torch
from tools import train, draw_loss, test, plot_confusion_matrix, print_statistic_information, draw_accuracy
import pandas as pd
from torch import nn, optim
from sklearn.model_selection import train_test_split
import os
from efficientnet_pytorch import EfficientNet
# pip install efficientnet_pytorch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set up the dataset path.
dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
path_df_trainset = os.path.join(dir_dataset, 'dataset-master/dataset-master/trainset.csv')
path_df_holdoutset = os.path.join(dir_dataset, 'dataset-master/dataset-master/holdoutset.csv')
df_trainset = pd.read_csv(path_df_trainset)
df_holdoutset = pd.read_csv(path_df_holdoutset)
df_trainset, df_valset = train_test_split(df_trainset, test_size = 0.1)



# Download the pretrained model.
# efficientnet-b3:
efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
num_ftrs = efficientnet._fc.in_features
efficientnet._fc = nn.Linear(num_ftrs, 4)

# Put the model in the GPU
efficientnet.to(DEVICE)


# Train the model
epochs = 20
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficientnet.parameters())

efficientnet, history = train(efficientnet, df_trainset, df_valset, epochs, loss_func, optimizer)

# Draw the loss graph
draw_loss(history)

# Draw the accuracy comparison graph
draw_accuracy(history)

# Test the accuracy on the holdout set
confusion_matirx = test(efficientnet, df_holdoutset)

# Plot the confusion matrix
classes = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
plot_confusion_matrix(confusion_matirx, classes)

# Print the statistic information:
print_statistic_information(confusion_matirx, classes)

# Put the model back to CPU
efficientnet.cpu()