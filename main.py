import torch
import matplotlib.pyplot as plt
from src.model import get_model
from src.train import train_model
from src.dataloader import get_dataloaders
from src.utils import get_loss
from src.config import DATA_PATH, BATCH_SIZE, IMAGE_SIZE, EPOCHS

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = get_model().to(device)
    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    loss_fn = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model and get loss history
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    # plot training and validation loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # inference preview
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images).squeeze()
        preds = (outputs > 0.5).float()

    print("Predictions:", preds.cpu().numpy())
    print("Ground truth:", labels.numpy())

if __name__ == "__main__":
    main()