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
path_df_valset = os.path.join(dir_dataset, 'dataset-master/dataset-master/valset.csv')
df_trainset = pd.read_csv(path_df_trainset)
df_holdoutset = pd.read_csv(path_df_holdoutset)
df_trainset, df_valset = train_test_split(df_trainset, test_size = 0.1)



# # Peek the image after preprocessing.
# tensor_image, label = next(iter(train_loader))['image'], next(iter(train_loader))['label'] # returns a batch of images
# print("Label of image", label[0])
# first_image = np.array(tensor_image, dtype='float')[0] # get the first image in the batch
# print(first_image.shape)
# #rerange the dimention to show the image.
# npimg = np.transpose(first_image,(1,2,0))
# plt.imshow(npimg)
# plt.show()

# Download the pretrained model.
# resnet 34:
resnet34 = models.resnet34(pretrained=True)
fc_inputs = resnet34.fc.in_features
resnet34.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Linear(10, 4),
    nn.LogSoftmax(dim=1)
)

# Put the model in the GPU
resnet34 = resnet34.cuda()

# Train the model
epochs = 20
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet34.parameters())
resnet34, history = train(resnet34, df_trainset, df_valset, epochs, loss_func, optimizer)

# Draw the loss graph
draw_loss(history)

# Draw the accuracy comparison graph
draw_accuracy(history)

# Test the accuracy on the holdout set
confusion_matirx = test(resnet34, df_holdoutset)

# Plot the confusion matrix
classes = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
plot_confusion_matrix(confusion_matirx, classes)

# Print the statistic information:
print_statistic_information(confusion_matirx, classes)

# Put the model back to CPU
resnet34.cpu()