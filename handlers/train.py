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

def get_model(config):
    model = ConvAE()
    return model

def save_model(model, config):
    model_state = {
        "num_epochs": config.get("num_epochs"),
        "encoder_dim": config.get("encoder_dim"),
        "decoder_dim": config.get("decoder_dim"),
        "state_dict": model.state_dict()
    }
    torch.save(model_state, config.get("saved_models_path")+config.get("executionName")+".pth")
def show_image(img):
    img = img.clamp(0, 1) # Ensure that the range of greyscales is between 0 and 1
    npimg = img.numpy()   # Convert to NumPy
    npimg = np.transpose(npimg, (2, 1, 0))   # Change the order to (W, H, C)
    plt.imshow(npimg)
    plt.show()

def train(model, loader, optimizer, criterion, reshape=False):
    loss = 0
    model.train()

    for batch_features, _ in loader:
        # load it to the active device
        batch_features = batch_features.to(device)

        # reshape mini-batch data to [N, 784] matrix (turn images into vectors, and subsume channel)
        if reshape:
            batch_features = batch_features.view(-1, 784)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = model(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(loader)
    print("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, epochs, loss))