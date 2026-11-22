import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_true, all_pred = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.numel()
        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())
    acc = correct / max(total, 1)
    loss_avg = loss_sum / max(total, 1)
    macro_f1 = float(f1_score(all_true, all_pred, average="macro")) if all_true else 0.0
    precision = float(precision_score(all_true, all_pred, average="macro", zero_division=0)) if all_true else 0.0
    recall = float(recall_score(all_true, all_pred, average="macro", zero_division=0)) if all_true else 0.0
    per_class_f1 = f1_score(all_true, all_pred, average=None, zero_division=0).tolist() if all_true else []
    return {
        "acc": acc,
        "loss": loss_avg,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "per_class_f1": per_class_f1,
    }
