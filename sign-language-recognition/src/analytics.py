import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from collections import Counter


def display_confusion_matrix(pred_sequences, ref_sequences):
    
    PAD_TOKEN = "<PAD>"

    y_true = []
    y_pred = []

    for ref, pred in zip(ref_sequences, pred_sequences):
        max_len = max(len(ref), len(pred))
        padded_ref = ref + [PAD_TOKEN] * (max_len - len(ref))
        padded_pred = pred + [PAD_TOKEN] * (max_len - len(pred))
        
        y_true.extend(padded_ref)
        y_pred.extend(padded_pred)
    
    labels = sorted(set(y_true + y_pred))
    print(len(labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, xticklabels=False, yticklabels=False, cmap='Blues', annot=False, cbar=True)

    plt.title("Confusion matrix")
    plt.xlabel("Prediction")
    plt.ylabel("References")
    plt.show()



def display_metrics(metrics):
    
    plt.plot([v.item() if torch.is_tensor(v) else v for v in metrics['val_loss']], color='orangered', label="Validation")
    plt.plot([v.item() if torch.is_tensor(v) else v for v in metrics['train_loss']], color='red', label="Train")
    plt.title("Loss evolution with epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")     
    plt.show()


def display_histogram(data, top_k=None, tick=False):
    glosses = data['gloss']
    
    # Count glosses
    all_glosses = [gloss for sequence in glosses for gloss in sequence]
    gloss_counts = Counter(all_glosses)

    # Top k selection 
    if top_k:
        most_common = gloss_counts.most_common(top_k)
    else:
        most_common = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
        
    glosses, counts = zip(*most_common)
    # Display histogram
    plt.figure(figsize=(12, 6))
    plt.bar(glosses, counts, color='dodgerblue')
    plt.title("Histogram")
    plt.xlabel("Gloss")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if tick:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks([], [])
    plt.show()
