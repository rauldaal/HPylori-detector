import os
import json
import logging
import wandb
import uuid

from handlers import (
    get_cropped_dataloader,
    get_annotated_dataloader,
    get_patients_dataloader,
    generate_model_objects,
    map_configuration,
    train,
    test,
    save_model,
    load_model,
    analyzer,
    compute_confussion_matrix,
    compute_classification,
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

            train_dataloader, validaiton_dataloader, used_patients = get_cropped_dataloader(config=config)
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
                model, criterion = load_model(config)
            all_true_labels, all_pred_labels, all_divisions = [], [], []
            true_labels, pred_labels, divisions = test(
                model=model,
                test_data_loader=pos_annotated_dataloader,
                criterion=criterion,
                label=1
            )
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            all_divisions.extend(divisions)
            true_labels, pred_labels, divisions = test(
                model=model,
                test_data_loader=neg_annotated_dataloader,
                criterion=criterion,
                label=-1
            )
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            all_divisions.extend(divisions)
            analyzer(results=all_pred_labels, true_labels=all_true_labels, project_path=config.get("project_path"))
            analyzer(results=all_divisions, true_labels=all_true_labels, project_path=config.get("project_path"))
            compute_confussion_matrix(true=all_true_labels, pred=all_pred_labels, project_path=config.get("project_path"))

            patients_dataloader, idx_patients, labels = get_patients_dataloader(config=config, used_patients=used_patients)
            compute_classification(dataloader=patients_dataloader, patients_idx=idx_patients, labels=labels, model=model, project_path=config.get("project_path"))

if __name__ == "__main__":
    with open("/fhome/mapsiv04/HPylori-detector/config.json", "r") as f:
        config_data = json.load(f)
        main(config=config_data)


