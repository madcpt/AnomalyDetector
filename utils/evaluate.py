import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_acc(outs):
    accs = []
    total_num = 0
    for pred, label in outs:
        pred = (pred >= 0.5).long().squeeze(1).cpu().numpy()
        label = label.cpu().numpy()
        accs.append(sum([int(pred[i] == label[i]) for i in range(len(pred))]))
        total_num += len(label)
    return float(sum(accs)) / total_num


def calculate_f1score(outs):
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in outs:
        pred = (pred >= 0.5).long().squeeze(1).cpu().numpy()
        label = label.cpu().numpy()
        for i in range(len(label)):
            if pred[i] == 1 and label[i] == 1:
                TP += 1
            elif pred[i] == 1 and label[i] == 0:
                FN += 1
            elif pred[i] == 0 and label[i] == 1:
                FP += 1
            elif pred[i] == 0 and label[i] == 0:
                TN += 1
            else:
                print(pred)
                print(label)
                raise  ValueError
    if TP == 0:
        return 0
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    return 2 * precision * recall / (precision + recall)