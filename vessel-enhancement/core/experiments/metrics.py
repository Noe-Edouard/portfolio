import torch
from numpy import ndarray
from monai.metrics import DiceMetric
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, roc_curve, precision_recall_curve


def dice(y_pred: ndarray, y_true: ndarray) -> float:
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        num_classes=2,
    )
    y_pred_t = torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0).float()
    y_true_t = torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0).float()

    return dice_metric(y_pred_t, y_true_t).item()

def mcc(y_pred: ndarray, y_true: ndarray) -> float:
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    return matthews_corrcoef(y_true, y_pred)

def roc(y_pred: ndarray, y_true: ndarray) -> float:
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    return roc_auc_score(y_true, y_pred)

def pr(y_pred: ndarray, y_true: ndarray) -> float:
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    return average_precision_score(y_true, y_pred)


