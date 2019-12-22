import os, sys

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

