from objects import ConvAE
import torch
import torchvision
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
from torchvision import datasets,transforms
import torchvision.transforms as transforms
#  use gpu if available
assert torch.cuda.is_available(), "GPU is not enabled"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import test
import train
from objects import model
import data

def kfold(batch_size,num_epochs,k):


    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()



    crop_data=data.generate_cropped_dataset()
    annotated_data=data.genearate_annotated_dataset()

    train_dataset,test_dataset=data.train_test_splitter(crop_data,0.2)
    dataset = ConcatDataset([train_dataset, test_dataset])

    # num_epochs=10
    # batch_size=128
    # k=10
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    foldperf={}

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        model = model.convAE()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        for epoch in range(num_epochs):
            train_loss, train_correct=train.train(model,train_loader,criterion,optimizer)
            test_loss, test_correct=test.test(model,test_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)