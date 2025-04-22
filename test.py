# test.py
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.eval import eval
from src.dataloader import get_dataloaders
from src.utils import get_loss
from src.config import DATA_PATH, BATCH_SIZE, IMAGE_SIZE

class OODDataset(Dataset):
    def __init__(self, folder_path, transform=None, target_label=0):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform
        self.target_label = target_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.target_label

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model architecture
    model = get_model().to(device)

    # Load weights
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()

    # Load test set
    _, _, test_loader = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    # Load AR generated imgs
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    ood_set = OODDataset(folder_path="ar_generated", transform=transform)
    ood_loader = DataLoader(ood_set, batch_size=BATCH_SIZE)
    loss_fn = get_loss()

    # Evaluate
    acc, auc, test_loss = eval(model, test_loader, loss_fn, device)
    acc_ood, auc_ood, test_loss_ood = eval(model, ood_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print(f"OOD Loss: {test_loss_ood:.4f}, Accuracy: {acc_ood:.4f}, AUC: {auc_ood:.4f}")

if __name__ == "__main__":
    main()