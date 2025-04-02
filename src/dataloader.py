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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    return train_loader, val_loader