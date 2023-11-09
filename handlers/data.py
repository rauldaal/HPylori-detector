from torch.utils.data import DataLoader, random_split
from objects import AnnotatedDataset, CroppedDataset


def get_cropped_dataloader(config):
    train, validation = generate_cropped_dataset(config)
    cropped_data_loader_train = DataLoader(
        dataset=train,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
        collate_fn=config.get("collateFn")
    )

    cropped_data_loader_validation = DataLoader(
        dataset=validation,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
        collate_fn=config.get("collateFn")
    )

    return cropped_data_loader_train, cropped_data_loader_validation


def get_annotated_dataloader(config):
    train, validation = genearate_annotated_dataset(config)
    annotated_data_loader_train = DataLoader(
        dataset=train,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
        collate_fn=config.get("collateFn")
    )
    annotated_data_loader_validation = DataLoader(
        dataset=validation,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
        collate_fn=config.get("collateFn")
    )

    return annotated_data_loader_train, annotated_data_loader_validation


def generate_cropped_dataset(config):
    cropped_dataset = CroppedDataset(
        folder_path=config.get("folder_path_cropped"),
        csv_name=config.get("cropped_csv")
    )
    return cropped_dataset


def genearate_annotated_dataset(config):
    annotated_dataset = AnnotatedDataset(
        folder_path=config.get("folder_path_annoted"),
        csv_name=config.get("annoted_csv")
    )
    return annotated_dataset


def train_test_splitter(dataset, split_value):
    train, test = random_split(dataset, [len(dataset)*split_value, len(dataset)*(1-split_value)])
    return train, test
