import torch
import torch.nn as nn
import torch.optim as optim

from ..objects import Autoencoder

assert torch.cuda.is_available(), "GPU is not enabled"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_model_objects(config):
    model = Autoencoder(**config).to(DEVICE)
    criterion = nn.MSELoss()
    if config.get("optimizer_type") == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate"))
    elif config.get("optimizer_type") == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config.get("learning_rate"), rho=0.95, eps=1e-07)
    elif config.get("optimizer_type") == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config.get("learning_rate"),
            lr_decay=config.get("lerning_reate_decay"),
            weight_decay=config.get("weight_decay"),
            initial_accumulator_value=0.1, eps=1e-10
            )

    return model, criterion, optimizer


def save_model(model, config):
    model_state = {
        'num_epochs': config.get("num_epochs"),
        'encoder_dim': config.get("encoder_dim"),
        'decoder_dim': config.get("decoder_dim"),
        'state_dict': config.get(model.state_dict()),
    }

    torch.save(model_state, config.get("executionName")+'.pth')

def load_model(config):
    model = Autoencoder(**config)
    model.load_state_dict(config.get("modelNmae"))
    model.eval()