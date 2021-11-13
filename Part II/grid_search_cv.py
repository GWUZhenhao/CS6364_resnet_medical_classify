from torchvision import models
from tools import sipit_cv, GridSearchCV, train, train_silence
import pandas as pd
from torch import nn, optim
import os
from sklearn.model_selection import train_test_split

# Set up the dataset path.
dir_dataset = 'D:/GWU/GWU Fall 2021/CS 6364/Group project/Dataset'
dir_row = os.path.join(dir_dataset, 'dataset-master/dataset-master')
dir_img = os.path.join(dir_row, 'JPEGImages')
path_df_trainset = os.path.join(dir_dataset, 'dataset-master/dataset-master/trainset.csv')
path_df_holdoutset = os.path.join(dir_dataset, 'dataset-master/dataset-master/holdoutset.csv')
df_trainset = pd.read_csv(path_df_trainset)
df_holdoutset = pd.read_csv(path_df_holdoutset)
df_trainset, _ = train_test_split(df_trainset, test_size = 0.1)

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


optimizer_1 = optim.Adam
optimizer_2 = optim.SGD

criterion_1 = nn.NLLLoss()
criterion_2 = nn.CrossEntropyLoss()
params = {
    'optimizer': [optimizer_1, optimizer_2],
    'lr': [0.01, 0.001],
    'criterion': [criterion_1, criterion_2]
}
cv_dataset = sipit_cv(df_trainset, 5)


best_param = GridSearchCV(resnet34, cv_dataset, params)






# # Testing code
# op = optim.Adam(resnet34.parameters())
# df_trainset = cv_dataset[0]['train']
# df_valset = cv_dataset[0]['validation']
# resnet34, history = train_silence(resnet34, df_trainset, df_valset, 10, criterion_1, op)



# # Another way to do the grid search, but still need time to study about it.
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
