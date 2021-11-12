import time
import copy
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from sklearn.metrics import confusion_matrix

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

# This function implement the gridsearchCV
def GridSearchCV(model, cv_dataset, params):
    optimizers = params['optimizer']
    epochs = params['epochs']
    criterions = params['criterion']
    accuracy_best = 0
    best_param = []
    index = 1
    print('Grid Searching begins:')
    for optim in optimizers:
        for num_epoch in epochs:
            for criterion in criterions:
                print('Try the {} hyper parameters:'.format(index))
                print('number of epochs = {}, optimizer = {}, criterion = {}'.format(num_epoch, optim, criterion))
                start = time.time()
                criterion = criterion()
                k_best, k_accuracy_best = CV_training(model, cv_dataset, num_epoch, criterion, optim)
                if k_accuracy_best > accuracy_best:
                    accuracy_best = k_accuracy_best
                    best_param = {'optimizer': optim, 'epochs': num_epoch, 'criterion': criterion}
                time_cost = time.time() - start
                print('Finished... Time consumption: {:.3f}'.format(time_cost))
                print('This CV best accuracy = {:.3f}'. format(k_accuracy_best))
                print()
                index += 1
    print('The grid search finished, the best hyper parameter is:')
    print('number of epochs = {}, optimizer = {}, criterion = {}'.format(best_param['epochs'], best_param['optimizer'], best_param['criterion']))
    return best_param

def sipit_cv(df_trainset, k):
    skf = StratifiedKFold(n_splits=k)
    df_feature = df_trainset.iloc[:,:-1]
    df_label = df_trainset.iloc[:,-1]
    cv_dataset = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_feature, df_label)):
        k_fold_dataset = {'train': df_trainset.iloc[train_idx,:], 'validation': df_trainset.iloc[val_idx,:]}
        cv_dataset.append(k_fold_dataset)
    return cv_dataset

def CV_training(model, cv_dataset, epochs, loss_func, optimizer):
    k_best = 0
    k_accuracy_best = 0
    for k, k_fold_datset in enumerate(cv_dataset):
        model_k = copy.deepcopy(model)
        df_trainset = k_fold_datset['train']
        df_valset = k_fold_datset['validation']
        k_model, k_history = train_silence(model_k, df_trainset, df_valset, epochs, loss_func, optimizer)
        k_accuracy = k_history['val_accuracy_best']
        if k_accuracy > k_accuracy_best:
            k_best = k
            k_accuracy_best = k_accuracy
    return k_best, k_accuracy_best

def train(model, df_trainset, df_valset, epochs, loss_func, optimizer):
    # Put the model in the GPU
    model = model.cuda()
    epochs_training_loss = np.array([], dtype='float64')
    epochs_val_loss = np.array([], dtype='float64')
    model_best = model
    loss_best = 1000
    epoch_best = 0
    train_accuracy = []
    val_accuracy = []
    history = {}

    # Generate the data loader.
    train_set = Dataset(df=df_trainset, transform=data_transform['train'])
    val_set = Dataset(df=df_valset, transform=data_transform['validation'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=16)

    for e in range(epochs):

        torch.cuda.empty_cache()
        # Calculate the time
        epoch_start = time.time()

        print("In epoch", e)
        train_running_loss = 0
        val_running_loss = 0

        model.train()

        right_predict = 0
        for batch in train_loader:  # processing one batch at a time

            images = batch['image']
            labels = batch['label']

            # Put the data into GPU
            images = images.cuda()
            labels = labels.cuda()

            predictions = model(images)  # predict labels
            loss = loss_func(predictions, labels)  # calculate the loss

            right_predict += torch.sum(torch.argmax(predictions, axis=1) == labels)

            # BACK PROPAGATION OF LOSS to generate updated weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_accuracy.append((right_predict / df_trainset.shape[0]).cpu().numpy())

        # Evaluate the model
        # Compute the evaluation loss
        model.eval()  # Turn off the model training, begin evaluate
        right_predict = 0
        for batch in validation_loader:
            images = batch['image']
            labels = batch['label']
            # Put the data into GPU
            images, labels = images.cuda(), labels.cuda()
            predictions = model(images)  # predict labels
            loss_val = loss_func(predictions, labels)  # calculate the loss
            right_predict += torch.sum(torch.argmax(predictions, axis=1) == labels)
            val_running_loss += loss_val.item()
        val_accuracy.append((right_predict / df_valset.shape[0]).cpu().numpy())

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
    history['loss_best'] = loss_best
    history['train_accuracy'] = train_accuracy
    history['val_accuracy'] = val_accuracy

    return(model_best, history)

def train_silence(model, df_trainset, df_valset, epochs, loss_func, optimizer):
    # Put the model in the GPU
    model = model.cuda()
    epochs_training_loss = np.array([], dtype='float64')
    epochs_val_loss = np.array([], dtype='float64')
    model_best = model
    loss_best = 1000
    epoch_best = 0
    train_accuracy = []
    val_accuracy = []
    history = {}

    # Generate the data loader.
    train_set = Dataset(df=df_trainset, transform=data_transform['train'])
    val_set = Dataset(df=df_valset, transform=data_transform['validation'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=16)

    for e in range(epochs):

        torch.cuda.empty_cache()
        # Calculate the time
        epoch_start = time.time()

        train_running_loss = 0
        val_running_loss = 0

        model.train()

        right_predict = 0
        for batch in train_loader:  # processing one batch at a time

            images = batch['image']
            labels = batch['label']

            # Put the data into GPU
            images = images.cuda()
            labels = labels.cuda()

            predictions = model(images)  # predict labels
            loss = loss_func(predictions, labels)  # calculate the loss

            right_predict += torch.sum(torch.argmax(predictions, axis=1) == labels)

            # BACK PROPAGATION OF LOSS to generate updated weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_accuracy.append(right_predict / df_trainset.shape[0])

        # Evaluate the model
        # Compute the evaluation loss
        model.eval()  # Turn off the model training, begin evaluate
        right_predict = 0
        for batch in validation_loader:
            images = batch['image']
            labels = batch['label']
            # Put the data into GPU
            images, labels = images.cuda(), labels.cuda()
            predictions = model(images)  # predict labels
            loss_val = loss_func(predictions, labels)  # calculate the loss
            right_predict += torch.sum(torch.argmax(predictions, axis=1) == labels)
            val_running_loss += loss_val.item()
        val_accuracy.append(right_predict / df_valset.shape[0])

        if val_running_loss / len(validation_loader) < loss_best:
            loss_best = val_running_loss / len(validation_loader)
            model_best = copy.deepcopy(model)
            epoch_best = e
            val_accuracy_best = right_predict / df_valset.shape[0]

        epochs_training_loss = np.append(epochs_training_loss, train_running_loss / len(train_loader))
        epochs_val_loss = np.append(epochs_val_loss, val_running_loss / len(validation_loader))

        epoch_end = time.time()

    history['training_loss'] = epochs_training_loss
    history['validation_loss'] = epochs_val_loss
    history['epoch_best'] = epoch_best
    history['loss_best'] = loss_best
    history['train_accuracy'] = train_accuracy
    history['val_accuracy'] = val_accuracy
    history['val_accuracy_best'] = val_accuracy_best

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

def draw_accuracy(history):
    plt.figure(figsize=(10, 7), facecolor='w')
    epochs_training_accuracy = history['train_accuracy']
    epochs_val_accuracy = history['val_accuracy']
    epochs = len(epochs_training_accuracy)
    plt.plot(np.array(range(epochs)) + 1., epochs_training_accuracy, 'o-b', label='train accuracy', lw=2)
    plt.plot(np.array(range(epochs)) + 1., epochs_val_accuracy, 'o-r', label='val accuracy', lw=2)
    plt.title('Accuracy graph')
    plt.legend()
    plt.show()

def test(model, df_holdoutset):
    model.eval()  # Turn off the model training, begin evaluate

    holdout_set = Dataset(df=df_holdoutset, transform=data_transform['validation'])
    holdout_loader = torch.utils.data.DataLoader(holdout_set, batch_size=len(holdout_set))

    data = next(iter(holdout_loader))
    image = data['image']
    label = data['label']
    image = image.cuda()
    prediction = model(image)
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    label = label.cpu().detach().numpy()
    num_right_predict = np.sum(prediction == label)
    num_predict = len(holdout_set)
    cm = confusion_matrix(label, prediction)

    print('The number of right prediction is {}'.format(num_right_predict))
    print('The number of total prediction is {}'.format(num_predict))
    print('The accuracy is {:.3f}'.format(num_right_predict / num_predict))
    return cm

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # Set the value for each cells
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.show()

def print_statistic_information(cm, classes):
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    print()
    print("Now, print the statistic_information")
    for index, type in enumerate(classes):
        print('For type {}:'.format(type))
        print('The recall/sensitivity is {:.3f}'.format(TPR[index]))
        print('The specificity is {:.3f}'.format(TNR[index]))
        print('The precision is {:.3f}'.format(PPV[index]))

    statistic_information = {'Sensitivity': TPR, 'Specificity': TNR, 'Precision': PPV, 'NPV': NPV, 'Fall_out': FPR, 'FNR': FNR, 'FDR': FDR}
    return statistic_information