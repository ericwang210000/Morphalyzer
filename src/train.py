from src.eval import eval
import torch
"""
    params:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        loss_compute (callable): Loss computation function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to perform training on.
    """

def train_model(model, train_loader, val_loader, epochs, loss_fn, optimizer, device):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    model = model.to(device)
    
    # iterate through epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # iterate through batches
        for batch in train_loader:
            # move batch to device
            images, targets = batch[0].to(device), batch[1].to(device).float()
            # forward pass
            logits = model(images)
            # compute loss
            loss = loss_fn(logits.squeeze(), targets)
            # zero gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            # accumulate loss
            total_loss += loss.item()

            # debug for first epoch
            """
            if epoch == 0:
                with torch.no_grad():
                    print("\n --- Epoch 1 Diagnostics ---")
                    print(f"Logits:\n{logits[:8].cpu().numpy()}")
                    probs = torch.sigmoid(logits)
                    print(f"Sigmoid probs:\n{probs[:8].cpu().numpy()}")
                    print(f"Targets:\n{targets[:8].cpu().numpy()}")
                    avg_conf = probs.mean().item()
                    print(f"Average confidence (sigmoid output): {avg_conf:.4f}")
                    print(f"Class 1 count: {(targets == 1).sum().item()} / {len(targets)}")
                    print("--------------------------------------------------\n")
            """

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # validation 
        acc, auc, avg_val_loss = eval(model, val_loader, loss_fn, device)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}, Accuracy: {acc}, AUC: {auc}")
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}")

        # save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"Best model saved with validation loss: {best_val_loss}")


    return train_losses, val_losses  
        