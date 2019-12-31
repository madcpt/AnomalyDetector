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


def evaluate_f1score_threshold(outs,bg=0.01,ed=0.6,step=0.01):
    thresholds = np.arange(bg+step, ed, step=step)
    outss = [[]] * len(thresholds)
    for t_id, t in enumerate(thresholds):
        for output, label in outs:
            pred = (output >= t).long()
            outss[t_id].append([pred, label])
    f1_scores = [calculate_f1score(outs) for outs in outss]
    best_t = thresholds[np.argmax(f1_scores)]
    return max(f1_scores)
#

def evaluate_acc(outs):
    accs = []
    total_num = 0
    for output, label in outs:
        pred = (output >=0.5).long()
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        accs.append(sum([int(pred[i] == label[i]) for i in range(len(pred))]))
        total_num += len(label)
    return float(sum(accs)) / total_num