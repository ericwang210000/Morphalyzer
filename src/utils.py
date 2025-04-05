import torch
import torch.nn.functional as F

# compute the binary cross-entropy loss using logits from NN and tensor-encoded "real" images
def loss_fn(logits, targets):
    loss_compute = torch.nn.BCEWithLogitsLoss()
    loss = loss_compute(logits, targets)
    return loss

def get_loss():
    return loss_fn

