import torch
import wandb
import tqdm


#assert torch.cuda.is_available(), "GPU is not enabled"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_data_loader, validation_data_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        print("+++++"*10)
        train_loss = 0

        model.train()
        for imgs in tqdm.tqdm(train_data_loader):
            imgs = imgs.to(DEVICE, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_data_loader)
        print("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, num_epochs, train_loss))

        validation_loss = 0
        model.eval()
        with torch.no_grad():
            for imgs in tqdm.tqdm(train_data_loader):
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                validation_loss += loss.item()
    
        validation_loss = validation_loss / len(validation_data_loader)
        print("EPOCH : {}/{}, Validation Loss = {:.6f}".format(epoch + 1, num_epochs, validation_loss))
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        wandb.log({"epoch": epoch, "validation_loss": validation_loss})

    return
