import torch
import wandb
import tqdm
import matplotlib as plt
import numpy as np
from torchvision.utils import make_grid
#  use gpu if available
#assert torch.cuda.is_available(), "GPU is not enabled"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  use gpu if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def show_image(img):
    img = img.clamp(0, 1) # Ensure that the range of greyscales is between 0 and 1
    npimg = img.numpy()   # Convert to NumPy
    npimg = np.transpose(npimg, (2, 1, 0))   # Change the order to (W, H, C)
    plt.imshow(npimg)
    plt.show()

def test(model, test_data_loader, criterion, label):
    if label == 1:
        label = 'positive'
    else:
        label = 'negative'
        
    columns=["id", "image", "predicted", "loss"]
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
            while num_images < max_num_images and i < len(outputs):
                single_loss = criterion(outputs[i], imgs[i])
                test_table.add_data(i, wandb.Image(imgs[i]), wandb.Image(outputs[i]), single_loss)
                i += 1
                num_images += 1
            # for i in range(min(max_num_images, len(outputs))):
            #     single_loss = criterion(outputs[i], imgs[i])
            #     test_table.add_data(i, wandb.Image(imgs[i]), wandb.Image(outputs[i]), single_loss)
    
    # compute the epoch test loss
    test_loss = test_loss / len(test_data_loader)
    
    # display the epoch training loss
    print(f"({label})Images Test loss = {test_loss:.6f}")
    wandb.log({f"{label} loss": test_loss})
    wandb.log({f"{label} predictions" : test_table})
    #show_image(make_grid(imgs.detach().cpu().view(-1, 1, 25, 25).transpose(2, 3), nrow=2, normalize = True))
    #show_image(make_grid(outputs.detach().cpu().view(-1, 1, 25, 25).transpose(2, 3), nrow=2, normalize = True)) 