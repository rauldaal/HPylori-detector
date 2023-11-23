import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score
)
from torchvision.utils import make_grid


#  use gpu if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 3.0


def test(model, test_data_loader, criterion, label):
    generated_labels = []
    divisions_calcs = []
    true_labels = []
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
        true_labels.extend(true_class)
        
        imgs = imgs.to(DEVICE, dtype=torch.float)

        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            test_loss += loss.item()
            wandb.log({"test_loss": test_loss})
            print("\nTest Loss", test_loss)
            i = 0
            ret, division = classifier(imgs, outputs)
            result = sum([x == y for x, y in zip(true_class, ret)]*1)
            generated_labels.extend(ret)
            divisions_calcs.extend(division)
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
    return true_labels, generated_labels, divisions_calcs


def convertir_a_hsv(input, output):
    input_results = []
    output_results = []
    input = input.permute(0, 2, 3, 1)
    input_canal_h = np.zeros_like(input[:, :, :, 0].to("cpu"), dtype=np.float32)

    output = output.permute(0, 2, 3, 1)
    output_canal_h = np.zeros_like(output[:, :, :, 0].to("cpu"), dtype=np.float32)

    # Itera sobre cada imagen en el batch
    for i in range(input.shape[0]):
        input_imagen_hsv = cv2.cvtColor(input[i].to("cpu").numpy(), cv2.COLOR_RGB2HSV)
        input_canal_h[i] = input_imagen_hsv[:, :, 0]
        f_red_input = np.sum(np.logical_or(input_canal_h[i] >= 340, input_canal_h[i] <= 20))
        #print("++++++"*5)
        #print(f"f_red_input:     {f_red_input}")

        output_imagen_hsv = cv2.cvtColor(output[i].to("cpu").numpy(), cv2.COLOR_RGB2HSV)
        output_canal_h[i] = output_imagen_hsv[:, :, 0]
        f_red_output = np.sum(np.logical_or(output_canal_h[i] >= 340, output_canal_h[i] <= 20))
        #print(f"f_red_output:     {f_red_output}")       
        #print("++++++"*5)
        input_results.append(f_red_input)
        output_results.append(f_red_output)

    return input_results, output_results


def classifier(input, output):
    
    batch_results = []
    division_results = []

    fred_input, fred_output = convertir_a_hsv(input, output)
    for fi, fo in zip(fred_input, fred_output):
        f = fi/fo if fo != 0 else fi
        division_results.append(f)
        if f > THRESHOLD:
            batch_results.append(1)
        else:
            batch_results.append(0)
    return batch_results, division_results


def analyzer(results, true_labels, project_path, name=None):
    fpr, tpr, thresholds = roc_curve(true_labels, results)
    roc_auc = auc(fpr, tpr)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]

    print(f'Umbral óptimo: {optimal_threshold}')
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    name = name if name else "roc.png"
    plt.savefig(project_path+"/plots/"+name)
    plt.show()
    return optimal_threshold


def compute_confussion_matrix(true, pred, project_path, name=None):
    plt.figure(figsize=(8, 8))
    conf_matrix = confusion_matrix(true, pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Reales')
    plt.title('Matriz de Confusión')
    name = name if name else "confussion.png"
    plt.savefig(project_path+"/plots/"+name)
    plt.show()
    acc = accuracy_score(true, pred)
    print(f"ACCURACY SCORE: {acc}")


def compute_classification(dataloader, patients_idx, labels, model, project_path):
    generated_labels = []

    for imgs in dataloader:
        imgs = imgs.to(DEVICE, dtype=torch.float)
        with torch.no_grad():
            outputs = model(imgs)
            ret, _ = classifier(imgs, outputs)

            generated_labels.extend(ret)
    
    labels_per_patient = {}
    print(len(list(generated_labels)))
    print(len(list(patients_idx.keys())))
    for indice, id_valor in patients_idx.items():
        if id_valor not in list(labels_per_patient.keys()):
            labels_per_patient[id_valor] = []
        labels_per_patient[id_valor].append(generated_labels[indice-1])
    
    probabilities = []
    actual_label = []

    for patient in labels_per_patient.keys():
        prob = sum(labels_per_patient[patient])/len(labels_per_patient[patient])
        probabilities.append(prob)
        actual_label.append(labels[patient])
    
    optimal = analyzer(results=probabilities, true_labels=actual_label, project_path=project_path, name="final_roc.png")


    final_results = []
    for patient in labels_per_patient.keys():
        prob = sum(labels_per_patient[patient])/len(labels_per_patient[patient])
        if prob > optimal:
            final_results.append(1)
        else:
            final_results.append(0)
    
    compute_confussion_matrix(true=actual_label, pred=final_results, project_path=project_path, name="final_cm.png")

