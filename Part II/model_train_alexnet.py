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
# Alexnet:
Alexnet = models.alexnet(pretrained=True)
Alexnet.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=4, bias=True)
)

# Put the model in the GPU
Alexnet = Alexnet.cuda()

# Train the model
epochs = 30
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(Alexnet.parameters())
# Using this optimizer to freeze some of the layers.
# optimizer_1 = optim.Adam([{'params':[ param for name, param in resnet34.named_parameters() if 'layer' in name]}], lr=0.1)


Alexnet, history = train(Alexnet, df_trainset, df_valset, epochs, loss_func, optimizer)

# Draw the loss graph
draw_loss(history)

# Draw the accuracy comparison graph
draw_accuracy(history)

# Test the accuracy on the holdout set
confusion_matirx = test(Alexnet, df_holdoutset)

# Plot the confusion matrix
classes = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
plot_confusion_matrix(confusion_matirx, classes)

# Print the statistic information:
print_statistic_information(confusion_matirx, classes)

# Put the model back to CPU
Alexnet.cpu()