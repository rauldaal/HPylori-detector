import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from objects import Autoencoder
import os
import io
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

    models_dir = os.path.join(config.get("project_path"), 'models')
    try:
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, config.get("execution_name") + '.pickle'), 'wb') as handle:
            pickle.dump(model, handle)
        print("Modelo Guardado Correctamente")
    except Exception as e:
        print(f"Error en el guardado. Error: {e}")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_model(name, config):

    models_dir = os.path.join(config.get("project_path"), 'models')
    with open(os.path.join(models_dir, config.get("model_name")+".pickle"), 'rb') as handle:
        # model = pickle.load(handle)
        model = CPU_Unpickler(handle).load()

    criterion = nn.MSELoss()
    return model, criterion
