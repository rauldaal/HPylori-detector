import os
import json
import logging
import wandb
import uuid

from handlers import (
    get_cropped_dataloader,
    get_annotated_dataloader,
    generate_model_objects,
    map_configuration,
    train,
    test,
    load_model
    )


def main(config):
    configurations = map_configuration(config_data=config)
    for config in configurations:
        if not config_data.get("execution_name"):
            config_data["executionName"] = config_data.get("projectName") + str(uuid.uuid4())[:-4]
        print(f"Configuration Parameters: {config}")
        wandb.init()
            # project=config.get("projectName"), name=config.get('execution_name'),
            # notes='execution', tags=['main'],
            # reinit=True, config=config)
            
            # wandb.define_metric('train_loss', step_metric='epoch')
            # wandb.define_metric('validation_loss', step_metric='epoch')

        # train_dataloader, validaiton_dataloader = get_cropped_dataloader(config=config)
        pos_annotated_dataloader, neg_annotated_dataloader = get_annotated_dataloader(config=config)
        # print('Cropped:')
        # print(f'Train batches: {len(train_dataloader)}')
        # i = 0
        # for imgs in train_dataloader:
        #     i += len(imgs)
        # print(f'Train num images: {i}')
        # print(f'Val batches: {len(validaiton_dataloader)}')
        # i = 0
        # for imgs in validaiton_dataloader:
        #     i += len(imgs)
        # print(f'Val num images: {i}')
        print('Annoted')
        print(f'Pos batches: {len(pos_annotated_dataloader)}')
        i = 0
        for imgs in pos_annotated_dataloader:
            i += len(imgs)
        print(f'Pos num images: {i}')
        print(f'Neg batches: {len(neg_annotated_dataloader)}')
        i = 0
        for imgs in neg_annotated_dataloader:
            i += len(imgs)
        print(f'Neg num images: {i}')
        
        # model, criterion, optimizer = generate_model_objects(config=config)

            # train(
            #     model=model,
            #     train_data_loader=train_dataloader,
            #     validation_data_loader=validaiton_dataloader,
            #     optimizer=optimizer,
            #     criterion=criterion,
            #     num_epochs=config.get("num_epochs"))
        no_use_model, criterion, optimizer = generate_model_objects(config=config)

        model=load_model("Hlbacter-detector7eea13f3-7c06-4a47-adb5-2dd7b6cb",config)
        test(
            model=model,
            test_data_loader=pos_annotated_dataloader,
            criterion=criterion,
            label=1
        )
        test(
            model=model,
            test_data_loader=neg_annotated_dataloader,
            criterion=criterion,
            label=-1
        )


if __name__ == "__main__":
    with open("/fhome/mapsiv04/HPylori-detector/config.json", "r") as f:
        config_data = json.load(f)
        main(config=config_data)


