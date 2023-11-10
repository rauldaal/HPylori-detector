import logging
import itertools
from functools import reduce

def map_configuration(config_data):
    """
    This funtion returns all the configuration option for multple executions
    Input:
        config_data: dict --> A dictionary containing all the configuration
    Returns:
        configurations: list(dicts) --> List of dictionaries containing all possible 
            combinations for multiple consecutive executions
    """
    MAPPED_VALUES = ["batchSize", "num_epochs", "image_size", "learning_rate", "optimizer_type"]

    if config_data.get("multiexecution"):

        batches = config_data.get("batchSize")
        epochs = config_data.get("num_epochs")
        image_sizes = config_data.get("image_size")
        learning_rates = config_data.get("learning_rate")
        optimizers = config_data.get("optimizer_type")
    
        values = [batches, epochs, image_sizes, learning_rates, optimizers]
        numcfg = reduce((lambda x, y: x * y), [len(i) for i in values])
        configurations = [{} for i in range(numcfg)]
        for iter in range(numcfg):
            for key, value in config_data.items():
                if key not in MAPPED_VALUES:
                    configurations[iter][key] = value
        combinations = list(itertools.product(batches, epochs, image_sizes, learning_rates, optimizers))

        for i, comb in enumerate(combinations):
            configurations[i][MAPPED_VALUES[0]] = comb[0]
            configurations[i][MAPPED_VALUES[1]] = comb[1]
            configurations[i][MAPPED_VALUES[2]] = comb[2]
            configurations[i][MAPPED_VALUES[3]] = comb[3]
            configurations[i][MAPPED_VALUES[4]] = comb[4]

        return configurations
    else:
        config_data["batchSize"] = config_data.get("batchSize")[0]
        config_data["num_epochs"] = config_data.get("num_epochs")[0]
        config_data["image_size"] = config_data.get("image_size")[0]
        config_data["learning_rate"] = config_data.get("learning_rate")[0]
        config_data["optimizer_type"] = config_data.get("optimizer_type")[0]
        
        return [config_data]
        
        

