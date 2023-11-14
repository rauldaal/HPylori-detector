import math
from torch import Generator
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from objects import AnnotatedDataset, CroppedDataset


def get_cropped_dataloader(config):
    dataset = generate_cropped_dataset(config)
    train, validation = train_test_splitter(dataset=dataset, split_value=0.8, seed=config.get("seed", 42))

    cropped_data_loader_train = DataLoader(
        dataset=train,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )

    cropped_data_loader_validation = DataLoader(
        dataset=validation,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )

    return cropped_data_loader_train, cropped_data_loader_validation


def get_annotated_dataloader(config):
    positive_dataset, negative_dataset = genearate_annotated_dataset(config)
    annotated_data_loader_pos = DataLoader(
        dataset=positive_dataset,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )
    annotated_data_loader_neg = DataLoader(
        dataset=negative_dataset,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )

    return annotated_data_loader_pos, annotated_data_loader_neg


def generate_cropped_dataset(config):
    cropped_dataset = CroppedDataset(
        folder_path=config.get("folder_path_cropped"),
        csv_name=config.get("cropped_csv"),
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.get("image_size"), config.get("image_size"))),
            transforms.ToTensor()
        ])
    )
    return cropped_dataset


def genearate_annotated_dataset(config):
    data = []
    for i in [1, -1]:
        annotated_dataset = AnnotatedDataset(
            folder_path=config.get("folder_path_annoted"),
            csv_name=config.get("annoted_csv"),
            #transform = transforms.Compose([
            #    transforms.ToPILImage(),
            #    transforms.Resize((config.get("image_size"), config.get("image_size"))),
            #    transforms.ToTensor()
            #]),
            label=i
        )
        data.append(annotated_dataset)
    return data[0], data[1]


def train_test_splitter(dataset, split_value, seed):
    size_train = math.ceil(len(dataset)*split_value)
    size_test = len(dataset)-size_train
    print(size_test, size_train)
    print(len(dataset))
    train, test = random_split(dataset, [size_train, size_test], generator=Generator().manual_seed(seed))
    return train, test
