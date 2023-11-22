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
    save_model,
    load_model
    )


def main(config):
    configurations = map_configuration(config_data=config)
    for config in configurations:
        if not config_data.get("execution_name"):
            config_data["executionName"] = config_data.get("projectName") + str(uuid.uuid4())[:-4]
        print(f"Configuration Parameters: {config}")
        with wandb.init(
            project=config.get("projectName"), name=config.get('execution_name'),
            notes='execution', tags=['main'],
            reinit = True, config=config):
            wandb.define_metric('train_loss', step_metric='epoch')
            wandb.define_metric('validation_loss', step_metric='epoch')

            train_dataloader, validaiton_dataloader = get_cropped_dataloader(config=config)
            pos_annotated_dataloader, neg_annotated_dataloader = get_annotated_dataloader(config=config)
            if not config.get("model_name"):

                model, criterion, optimizer = generate_model_objects(config=config)

                train(
                    model=model,
                    train_data_loader=train_dataloader,
                    validation_data_loader=validaiton_dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    num_epochs=config.get("num_epochs"))
                save_model(model, config)
            else:
                model, criterion = load_model(config.get("model_name"), config)
            true_labels_pos,pred_labels_pos=test(
                model=model,
                test_data_loader=pos_annotated_dataloader,
                criterion=criterion,
                label=1
            )
            true_labels_neg,pred_labels_neg=test(
                model=model,
                test_data_loader=neg_annotated_dataloader,
                criterion=criterion,
                label=-1
            )
            true_labels_pos.extend(true_labels_neg)
            pred_labels_pos.extend(pred_labels_neg)


            print(f"Final True Labels: {true_labels_pos}")
            print(f"Final Pred Labels: {pred_labels_pos}")
            print(len(true_labels_pos),len(pred_labels_pos))

            # plt.figure()
            # fpr,tpr,thresholds=roc_curve(true_labels_pos,pred_labels_pos)
            # plt.plot(fpr,tpr,marker=".")
            # plt.show()
            wandb.log({f"roc" : wandb.plot.roc_curve(true_labels_pos, pred_labels_pos,labels=None,classes_to_plot=None)})


if __name__ == "__main__":
    with open("/fhome/mapsiv04/HPylori-detector/config.json", "r") as f:
        config_data = json.load(f)
        main(config=config_data)


