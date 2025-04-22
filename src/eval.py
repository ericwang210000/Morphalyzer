import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# src/evaluate.py
def eval(model, dataloader, loss_fn, device, is_binary=True):
    # set model to eval mode
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    # 
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device).float()
            outputs = model(images).squeeze()

            if is_binary:
                preds = (outputs > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1)

            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds) if is_binary else None
    return acc, auc, total_loss / len(dataloader)