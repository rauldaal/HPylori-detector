import cv2
import matplotlib as plt
import numpy as np
import torch
import tqdm
import wandb

from torchvision.utils import make_grid


#  use gpu if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_data_loader, criterion, label):
    if label == 1:
        label = 'positive'
    else:
        label = 'negative'

    columns = ["id", "image", "predicted", "loss"]
    test_table = wandb.Table(columns=columns)
    max_num_images = 10
    num_images = 0

    test_loss = 0
    model.eval()
    print("++++++++"*10)
    for imgs in tqdm.tqdm(test_data_loader):
        imgs = imgs.to(DEVICE, dtype=torch.float)

        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            test_loss += loss.item()
            wandb.log({"test_loss": test_loss})
            print("\nTest Loss", test_loss)
            i = 0
            ret = classifier(imgs, outputs)
            while num_images < max_num_images and i < len(outputs):
                single_loss = criterion(outputs[i], imgs[i])
                test_table.add_data(i, wandb.Image(imgs[i]), wandb.Image(outputs[i]), single_loss)
                i += 1
                num_images += 1

    # compute the epoch test loss
    test_loss = test_loss / len(test_data_loader)

    # display the epoch training loss
    print(f"({label})Images Test loss = {test_loss:.6f}")
    wandb.log({f"{label} loss": test_loss})
    wandb.log({f"{label} predictions": test_table})


def convertir_a_hsv(batch):
    batch = batch.permute(0, 2, 3, 1)
    batch_canal_h = np.zeros_like(batch[:, :, :, 0], dtype=np.float32)

    # Itera sobre cada imagen en el batch
    for i in range(batch.shape[0]):
        imagen_hsv = cv2.cvtColor(batch[i].numpy(), cv2.COLOR_BGR2HSV)
        batch_canal_h[i] = imagen_hsv[:, :, 0]

    return batch_canal_h


def classifier(input, output):
    input_h = convertir_a_hsv(input)
    output_h = convertir_a_hsv(output)
    input_values = np.logical_and(input_h >= -20, input_h <= 20)
    num_input = np.sum(input_values)
    output_values = np.logical_and(output_h >= -20, output_h <= 20)
    num_output = np.sum(output_values)

    res = num_input/num_output
    print(f"Numero red entrada: {num_input}")
    print(f"Numero red salida: {num_output}")
    print(f"Frecuencia:  {res}")
    return res

