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
    label_n=label
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
    true_labels=[]
    pred_labels=[]
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
            pred_batch = classifier(imgs, outputs)
            # print(ret,label)
            
            # pred_batch = [1 if x > 1 else -1 for x in ret]
            # print(pred_batch)
            # print(len(ret),len(pred_batch))
            if label_n == 1:
                true_batch = [1] * len(pred_batch)
            else:
                true_batch = [-1] * len(pred_batch)
            
            true_labels.extend(true_batch)
            pred_labels.extend(pred_batch)

            while num_images < max_num_images and i < len(outputs):
                single_loss = criterion(outputs[i], imgs[i])
                test_table.add_data(i, wandb.Image(imgs[i]), wandb.Image(outputs[i]), single_loss)
                i += 1
                num_images += 1

    # compute the epoch test loss
    test_loss = test_loss / len(test_data_loader)

    # display the epoch training loss

    print(f"True Labels: {true_labels}")
    print(f"Pred Labels: {pred_labels}")
    print(len(true_labels),len(pred_labels))
    print(f"({label})Images Test loss = {test_loss:.6f}")
    wandb.log({f"{label} loss": test_loss})
    wandb.log({f"{label} predictions": test_table})

    return true_labels,pred_labels




def convertir_a_hsv(batch):
    batch = batch.permute(0, 2, 3, 1)
    batch_canal_h = np.zeros_like(batch[:, :, :, 0].to('cpu'), dtype=np.float32)

    # Itera sobre cada imagen en el batch
    for i in range(batch.shape[0]):
        #img = cv2.cvtColor(batch[i].cpu().numpy(), cv2.COLOR_BGR2RGB)
        imagen_hsv = cv2.cvtColor(batch[i].cpu().numpy(), cv2.COLOR_BGR2HSV)
        batch_canal_h[i] = imagen_hsv[:, :, 0]

    return batch_canal_h


def classifier(input, output):
    input_h = convertir_a_hsv(input)
    output_h = convertir_a_hsv(output)
    ret=[]
    for i in range(output_h.shape[0]):
        input_values = np.logical_and(input_h[i] >= -15, input_h[i] <= 15)
        num_input = np.sum(input_values)
        output_values = np.logical_and(output_h[i] >= -15, output_h[i] <= 15)
        num_output = np.sum(output_values)
        res = num_output/num_input
        ret.append(res)
        # print(f"Numero red entrada: {num_input}")
        # print(f"Numero red salida: {num_output}")
        # print(f"Frecuencia:  {res}")
    return ret

