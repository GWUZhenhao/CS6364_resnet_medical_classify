from torchvision import models
from tools import sipit_cv, GridSearchCV
import pandas as pd
from torch import nn, optim
import os

# Set up the dataset path.
dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
path_df_trainset = os.path.join(dir_dataset, 'dataset-master/dataset-master/trainset.csv')
path_df_holdoutset = os.path.join(dir_dataset, 'dataset-master/dataset-master/holdoutset.csv')
df_trainset = pd.read_csv(path_df_trainset)
df_holdoutset = pd.read_csv(path_df_holdoutset)

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



optimizer_1 = optim.SGD(resnet34.parameters(), 0.01)
optimizer_2 = optim.Adam(resnet34.parameters())
optimizer_3 = optim.Adam([{'params':[ param for name, param in resnet34.named_parameters() if 'layer' in name]}], lr=0.1)
params = {
    'optimizer': [optimizer_1, optimizer_2, optimizer_3],
    'epochs': [20],
    'criterion': [nn.NLLLoss, nn.CrossEntropyLoss]
}
cv_dataset = sipit_cv(df_trainset, 5)
best_param = GridSearchCV(resnet34, cv_dataset, params)











# from skorch import NeuralNetRegressor
# net = NeuralNetRegressor(resnet34, optimizer=optim.Adam, criterion=nn.NLLLoss, verbose=1, device='cuda')
# gs = GridSearchCV(net, params, refit=False, scoring='r2', verbose=1, cv=10)
# train_set = Dataset(df=df_trainset, transform=data_transform['train'])
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
# images = torch.empty([len(train_loader), 3, 224, 224])
# labels = torch.empty([len(train_loader), 1])
# for index, data in enumerate(train_loader):
#     image = data['image']
#     label = data['label']
#     images[index] = image[0]
#     labels[index] = label
# # net.fit(train_set, y=None)
# gs.fit(images, labels)
