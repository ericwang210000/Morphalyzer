import torch

def train_model(model, train_loader, val_loader, epochs, loss_fn, optimizer, device):
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
    model = model.to(device)
    
    # iterate through epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # iterate through batches
        for batch in train_loader:
            # move batch to device
            images, targets = batch[0].to(device), batch[1].to(device)
            # forward pass
            logits = model(images)
            # compute loss
            loss = loss_fn(logits, targets)
            # zero gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            # accumulate loss
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
