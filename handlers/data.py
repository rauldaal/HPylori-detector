from torch.utils.data import DataLoader, random_split
from ..objects import AnnotatedDataset, CroppedDataset


def get_cropped_dataloader(config):
    cropped_data_loader = DataLoader(
        dataset=generate_cropped_dataset(config),
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
        collate_fn=config.get("collateFn")
    )

    return cropped_data_loader


def get_annotated_dataloader(config):
    annotated_data_loader = DataLoader(
        dataset=genearate_annotated_dataset(config),
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
        collate_fn=config.get("collateFn")
    )

    return annotated_data_loader


def generate_cropped_dataset(config):
    cropped_dataset = CroppedDataset(
        folder_path=config.get("folder_path"),
        csv_name=config.get("csv_name")
    )
    return cropped_dataset


def genearate_annotated_dataset(config):
    annotated_dataset = AnnotatedDataset(
        folder_path=config.get("folder_paht"),
        csv_name=config.get("csv_name")
    )
    return annotated_dataset


def train_test_splitter(dataset, split_value):
    train, test = random_split(dataset, [len(dataset)*split_value, len(dataset)*(1-split_value)])
    return train, test
