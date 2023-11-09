import logging

def map_configuration(config_data):
    """
    This funtion returns all the configuration option for multple executions
    Input:
        config_data: dict --> A dictionary containing all the configuration
    Returns:
        configurations: list(dicts) --> List of dictionaries containing all possible 
            combinations for multiple consecutive executions
    """
    MAPPED_VALUES = ["batchSize", "num_epochs", "encoder_dim", "image_size"] # More to be added...
