import torch
from torch.utils.data import DataLoader

NUM_WORKERS = 1


def data_loader_func(dataset, batch_size, device, shuffle=True):
    if 0 < batch_size < len(dataset):
        return DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=shuffle)
    else:
        return [[item.to(device) for item in items] for items in
                list(DataLoader(dataset, batch_size=len(dataset), num_workers=NUM_WORKERS))]


def get_data_loaders(dataset, batch_size, device):
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    train_data_loader = data_loader_func(train_dataset, batch_size, device, shuffle=True)
    valid_data_loader = data_loader_func(valid_dataset, batch_size, device, shuffle=False)
    test_data_loader = data_loader_func(test_dataset, batch_size, device, shuffle=False)
    return train_data_loader, valid_data_loader, test_data_loader


def get_data_loader(dataset, batch_size, device):
    return data_loader_func(dataset, batch_size, device)
