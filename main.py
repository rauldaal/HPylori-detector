import os
import json
import logging
import wandb
import uuid



def main(config):
    with wandb.init(
        project=config.get("projectName"), name=config.get('execution_name'),
        notes='execution', tags=['main'],
        reinit=True, config=config):
        wandb.define_metric('loss_train', step_metric='epoch')
        wandb.define_metric('loss_test', step_metric='epoch')


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config_data = json.load(f)
        if not config_data.get("execution_name"):
            config_data["executionName"] = config_data.get("projectName") + str(uuid.uuid4())[:-4]
        main(config=config_data)


