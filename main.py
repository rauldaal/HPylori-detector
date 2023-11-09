import os
import json
import logging
import wandb
import uuid

from handlers import (
    get_cropped_dataloader,
    generate_model_objects,
    train
    )



def main(config):
    with wandb.init(
        project=config.get("projectName"), name=config.get('execution_name'),
        notes='execution', tags=['main'],
        reinit=True, config=config):
        wandb.define_metric('train_loss', step_metric='epoch')
        wandb.define_metric('validation_loss', step_metric='epoch')

        train_dataloader, validaiton_dataloader = get_cropped_dataloader(config=config)

        model, optimizer, criterion = generate_model_objects(config=config)
        train(
            model=model,
            train_data_loader=train_dataloader,
            validation_data_loader=validaiton_dataloader,
            optimizer=optimizer,
            criterion=criterion)







if __name__ == "__main__":
    with open("config.json", "r") as f:
        config_data = json.load(f)
        if not config_data.get("execution_name"):
            config_data["executionName"] = config_data.get("projectName") + str(uuid.uuid4())[:-4]
        main(config=config_data)


