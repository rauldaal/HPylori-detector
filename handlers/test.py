import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb

from torchvision.utils import make_grid


#  use gpu if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 1.9


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
        if label=="positive":
            true_class = [1 for i in range(imgs.shape[0])]
        elif label == "negative" :
            true_class = [0 for i in range(imgs.shape[0])]
        
        imgs = imgs.to(DEVICE, dtype=torch.float)

        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            test_loss += loss.item()
            wandb.log({"test_loss": test_loss})
            print("\nTest Loss", test_loss)
            i = 0
            ret = classifier(imgs, outputs)
            result = sum([x == y for x, y in zip(true_class, ret)]*1)
            print("++++++++"*10)
            print(F"BATCH RESULT {label}:  {result}/{len(ret)}")

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


def convertir_a_hsv(input, output):
    input_results = []
    output_results = []
    input = input.permute(0, 2, 3, 1)
    input_canal_h = np.zeros_like(input[:, :, :, 0], dtype=np.float32)

    output = output.permute(0, 2, 3, 1)
    output_canal_h = np.zeros_like(output[:, :, :, 0], dtype=np.float32)

    # Itera sobre cada imagen en el batch
    for i in range(input.shape[0]):
        input_imagen_hsv = cv2.cvtColor(input[i].numpy(), cv2.COLOR_RGB2HSV)
        input_canal_h[i] = input_imagen_hsv[:, :, 0]
        f_red_input = np.sum(np.logical_or(input_canal_h[i] >= 340, input_canal_h[i] <= 20))
        #print("++++++"*5)
        #print(f"f_red_input:     {f_red_input}")

        output_imagen_hsv = cv2.cvtColor(output[i].numpy(), cv2.COLOR_RGB2HSV)
        output_canal_h[i] = output_imagen_hsv[:, :, 0]
        f_red_output = np.sum(np.logical_or(output_canal_h[i] >= 340, output_canal_h[i] <= 20))
        #print(f"f_red_output:     {f_red_output}")       
        #print("++++++"*5)
        input_results.append(f_red_input)
        output_results.append(f_red_output)

    return input_results, output_results


def classifier(input, output):
    
    batch_results = []

    fred_input, fred_output = convertir_a_hsv(input, output)
    for fi, fo in zip(fred_input, fred_output):
        f = fi/fo if fo != 0 else fi
        if f > THRESHOLD:
            batch_results.append(1)
        else:
            batch_results.append(0)
    return batch_results

