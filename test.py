# test.py
import torch
from src.model import get_model
from src.eval import eval
from src.dataloader import get_dataloaders
from src.utils import get_loss
from src.config import DATA_PATH, BATCH_SIZE, IMAGE_SIZE

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model architecture
    model = get_model().to(device)

    # Load weights
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()

    # Load test set
    _, _, test_loader = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    loss_fn = get_loss()

    # Evaluate
    acc, auc, test_loss = eval(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    main()