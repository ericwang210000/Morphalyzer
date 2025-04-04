import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import IMAGE_SIZE, BATCH_SIZE

def get_dataloaders(data_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # First split: Train vs temp (val + test)
    train_size = int(0.7 * len(dataset))
    temp_size = len(dataset) - train_size
    train_set, temp_set = torch.utils.data.random_split(dataset, [train_size, temp_size])

    # Second split: Validation vs Test (from temp)
    val_size = int(0.5 * len(temp_set))
    test_size = len(temp_set) - val_size
    val_set, test_set = torch.utils.data.random_split(temp_set, [val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader