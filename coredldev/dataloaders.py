from .dataset import *
import torch.utils.data as data
import numpy as np


def train_validation_test_dataloaders(
    dataset: CoReDataset,
    train_batch_size=16,
    validation_batch_size=16,
    test_batch_size=16,
    validation_split=0.1,
    test_split=0.1,
    shuffle_dataset=True,
):
    source = dataset.source
    preprocessor = dataset.preprocessor
    sample_list = dataset.sample_list
    np.random.shuffle(sample_list)
    length = len(sample_list)
    test_endpoint = int(length * test_split)
    validation_endpoint = int(length * (test_split + validation_split))
    test_datapoints = sample_list[:test_endpoint]
    validation_datapoints = sample_list[test_endpoint:validation_endpoint]
    train_datapoints = sample_list[validation_endpoint:]
    print(len(test_datapoints), len(validation_datapoints), len(train_datapoints))
    train_dataset = CoReDataset(
        source=source, sample_list=train_datapoints, preprocessor=preprocessor
    )
    validation_dataset = CoReDataset(
        source=source, sample_list=validation_datapoints, preprocessor=preprocessor
    )
    test_dataset = CoReDataset(
        source=source, sample_list=test_datapoints, preprocessor=preprocessor
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=shuffle_dataset
    )
    validation_dataloader = data.DataLoader(
        dataset=validation_dataset,
        batch_size=validation_batch_size,
        shuffle=shuffle_dataset,
    )
    test_dataloader = data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=shuffle_dataset
    )

    return train_dataloader, validation_dataloader, test_dataloader
