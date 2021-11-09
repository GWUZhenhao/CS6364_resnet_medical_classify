import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df=None, transform=None):
        self.df = df
        self.df_len = df.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Read image and do the image augumentation
        path_img = self.df.iloc[index, 0]
        img = Image.open(path_img)
        if self.transform:
            img = self.transform(img)

        # Read label
        label = self.df.iloc[index, 1]
        if label == 'NEUTROPHIL':
            label = 0
        elif label == 'EOSINOPHIL':
            label = 1
        elif label == 'MONOCYTE':
            label = 2
        elif label == 'LYMPHOCYTE':
            label = 3
        else:
            raise Exception('Error: invalid class, index = {}'.format(index))

        sample = {'image': img, 'label': label}
        return sample

    def __len__(self):
        return self.df_len

def train(model, train_loader, validation_loader, epochs, loss_func, optimizer):

    epochs_training_loss = np.array([], dtype='float64')
    epochs_val_loss = np.array([], dtype='float64')
    model_best = model
    loss_best = 1000
    epoch_best = 0
    history = {}

    for e in range(epochs):

        torch.cuda.empty_cache()
        # Calculate the time
        epoch_start = time.time()

        print("In epoch", e)
        train_running_loss = 0
        val_running_loss = 0

        model.train()

        for batch in train_loader:  # processing one batch at a time

            images = batch['image']
            labels = batch['label']

            # Put the data into GPU
            images = images.cuda()
            labels = labels.cuda()

            predictions = model(images)  # predict labels
            loss = loss_func(predictions, labels)  # calculate the loss

            # BACK PROPAGATION OF LOSS to generate updated weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        # Evaluate the model
        # Compute the evaluation loss
        model.eval()  # Turn off the model training, begin evaluate
        for batch in validation_loader:
            images = batch['image']
            labels = batch['label']
            # Put the data into GPU
            images, labels = images.cuda(), labels.cuda()
            predictions = model(images)  # predict labels
            loss_val = loss_func(predictions, labels)  # calculate the loss
            val_running_loss += loss_val.item()

        if val_running_loss / len(validation_loader) < loss_best:
            loss_best = val_running_loss / len(validation_loader)
            model_best = copy.deepcopy(model)
            epoch_best = e

        epochs_training_loss = np.append(epochs_training_loss, train_running_loss / len(train_loader))
        epochs_val_loss = np.append(epochs_val_loss, val_running_loss / len(validation_loader))

        epoch_end = time.time()
        print('\tTime consumption: {:.4f}'.format(epoch_end - epoch_start))
        print('\tTraining loss: {:.4f}'.format(train_running_loss / len(train_loader)))
        print('\tValidation loss: {:.4f}'.format(val_running_loss / len(validation_loader)))

    history['training_loss'] = epochs_training_loss
    history['validation_loss'] = epochs_val_loss
    history['epoch_best'] = epoch_best
    return(model_best, history)

def draw_loss(history):
    plt.figure(figsize=(10, 7), facecolor='w')
    epochs_training_loss = history['training_loss']
    epochs_val_loss = history['validation_loss']
    epochs = len(epochs_training_loss)
    plt.plot(np.array(range(epochs)) + 1., epochs_training_loss, 'o-b', label='train loss', lw=2)
    plt.plot(np.array(range(epochs)) + 1., epochs_val_loss, 'o-r', label='val loss', lw=2)
    plt.title('Loss graph')
    plt.legend()
    plt.show()

def test(model, holdout_loader):
    model.eval()  # Turn off the model training, begin evaluate
    num_right_predict = 0
    num_predict = 0

    for data in holdout_loader:
        image = data['image']
        label = data['label']
        image = image.cuda()
        prediction = model(image)
        prediction = prediction.cpu().detach().numpy()
        if np.argmax(prediction) == label:
            num_right_predict += 1
        num_predict += 1

    print('The number of right prediction is {}'.format(num_right_predict))
    print('The number of total prediction is {}'.format(num_predict))
    print('The accuracy is {}'.format(num_right_predict / num_predict))