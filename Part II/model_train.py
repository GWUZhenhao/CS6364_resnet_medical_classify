from torchvision import transforms
from tools import Dataset, train, draw_loss, test
import torch
import pandas as pd
from torch import nn, optim
import os


# Define data_transform
data_transform = {'train': transforms.Compose([transforms.RandomRotation(degrees=90),
                                       transforms.ColorJitter(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.Resize(size=[224,224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ]),
                  'validation': transforms.Compose([transforms.RandomRotation(degrees=90),
                                       transforms.ColorJitter(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.Resize(size=[224,224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ]),
                  'holdout': transforms.Compose([transforms.RandomRotation(degrees=90),
                                       transforms.ColorJitter(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.Resize(size=[224,224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
                 }

# Set up the dataset path.
dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
path_df_trainset = os.path.join(dir_dataset, 'dataset-master/dataset-master/trainset.csv')
path_df_holdoutset = os.path.join(dir_dataset, 'dataset-master/dataset-master/holdoutset.csv')
path_df_valset = os.path.join(dir_dataset, 'dataset-master/dataset-master/valset.csv')
df_trainset = pd.read_csv(path_df_trainset)
df_holdoutset = pd.read_csv(path_df_holdoutset)
df_valset = pd.read_csv(path_df_valset)

# Generate the data loader.
train_set = Dataset(df=df_trainset, transform=data_transform['train'])
val_set = Dataset(df=df_valset, transform=data_transform['validation'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
validation_loader = torch.utils.data.DataLoader(val_set, batch_size=16)

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
from torchvision import models
resnet50 = models.resnet50(pretrained=True)
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.ReLU(), # add Relu function for each linear layers.
    nn.Linear(10, 4),
    nn.LogSoftmax(dim=1)
)

# Put the model in the GPU
resnet50 = resnet50.cuda()
# Train the model
epochs = 10
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())
resnet50, history = train(resnet50, train_loader, validation_loader, epochs, loss_func, optimizer)

# Draw the loss graph
draw_loss(history)

# Test the accuracy on the holdout set
holdout_set = Dataset(df=df_holdoutset, transform=data_transform['validation'])
holdout_loader = torch.utils.data.DataLoader(holdout_set, batch_size=1)
test(resnet50, holdout_loader)