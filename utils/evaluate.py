import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_acc(outs):
    accs = []
    total_num = 0
    for pred, label in outs:
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        accs.append(sum([int(pred[i] == label[i]) for i in range(len(pred))]))
        total_num += len(label)
    return float(sum(accs)) / total_num


def calculate_f1score(outs):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, y in outs:
        tp += ((pred == y) & (pred == 1)).sum().item()
        tn += ((pred == y) & (pred != 1)).sum().item()
        fp += ((pred != y) & (pred == 1)).sum().item()
        fn += ((pred != y) & (pred != 1)).sum().item()
    if tp == 0:
        return 0
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def evaluate_f1score_threshold(outs, bg=0, ed=0.6, step=0.06):
    thresholds = np.arange(bg + step, ed, step=step)
    f1_scores = []
    for t in thresholds:
        preds = []
        for output, label in outs:
            pred = (output >= t).long()
            preds.append([pred, label])
        f1_scores.append(calculate_f1score(preds))
    # print(f1_scores)
    best_t = thresholds[np.argmax(f1_scores)]
    return max(f1_scores)


#

def evaluate_acc(outs):
    accs = []
    total_num = 0
    for output, label in outs:
        pred = (output >= 0.5).long()
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        accs.append(sum([int(pred[i] == label[i]) for i in range(len(pred))]))
        total_num += len(label)
    return float(sum(accs)) / total_num


import torch
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score


def normalize_and_get_f1_score(labels, scores, bg=0.05, ed=0.90, step=0.05):
    # print(scores)
    scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
    thresholds = np.arange(bg, ed, step=step)
    f1_scores = []
    for threshold in thresholds:
        scores_ = (scores < threshold).long().cpu()
        f1_scores.append(f1_score(labels, scores_))
    return max(f1_scores)
