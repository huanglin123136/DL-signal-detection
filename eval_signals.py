import os
import numpy as np

def iou_g(gt, loc, label):
    gt_label = gt['label']
    gt_loc = gt['loc']
    loc_min = max(loc[0], gt_loc[0])
    loc_max = min(loc[1], gt_loc[1])
    inter = loc_max - loc_min
    iou = inter / (loc[1] - loc[0] + gt_loc[1] - gt_loc[0] - inter)
    if iou < 0.2 or label != gt_label:
        return False
    else:
        return True

def eval_sample(path):
    with open(path, 'r') as f:
        data = f.read()
    blocks = data.split('Groud ')[1:]
    fp_blob = []
    tp_blob = []
    fn_blob = []
    for block in blocks:
        gt_block = block.split('Prediction:')[0]
        gt_lines = gt_block.split('\n')[1:-1]
        gt_lines = [x.strip() for x in gt_lines]
        gt = []
        for line in gt_lines:
            gt_label = {}
            line = line.split(' ')[-1]
            label = line.split('||')
            gt_label['label'] = int(label[-1])
            gt_label['loc'] = np.array(label[:-1], dtype=np.float)
            gt.append(gt_label)
        bbox_block = block.split('Prediction:')[1]
        bbox_lines = bbox_block.split('\n')[1:-2]
        scores = []
        locs = []
        classes = []
        for line in bbox_lines:
            if line.find('AM') != -1:
                classes.append(0)
            elif line.find('DSB') != -1:
                classes.append(1)
            bbox_line = line.split(':')[-1].split(' ')
            score = float(bbox_line[-2])
            scores.append(score)
            loc = np.array(bbox_line[-1].split('||'), dtype=np.float)
            locs.append(loc)

        #matching
        det = [False] * len(gt)
        scores = np.array(scores)
        sort_index = np.argsort(-scores)
        sorted_scores = [scores[x] for x in sort_index]
        sorted_locs = [locs[x] for x in sort_index]
        sorted_classes = [classes[x] for x in sort_index]
        tp = 0
        fp = 0
        matched_gt = 0
        matched_times = 0
        for i in range(len(sort_index)):
            pred_loc = sorted_locs[i]
            pred_label = sorted_classes[i]
            match = 0
            matched_times += 1
            for j in range(len(gt)):
                if match == 0 and  det[j] == False and iou_g(gt[j], pred_loc, pred_label):
                    det[j] = True
                    match = 1
                    matched_gt += 1
                    tp += 1
            if match == 0:
                fp += 1
            if matched_gt == len(gt):
                break

        fn_blob.append((len(gt) - matched_gt) / len(gt))
        fp_blob.append(fp/matched_times if matched_times != 0 else 0)
        tp_blob.append(tp/matched_times if matched_times != 0 else 0)
    return tp_blob, fp_blob, fn_blob

def eval_signals(path):
    tp, fp, fn = eval_sample(path)
    wrong_1 = [abs(x - 1.0) < 0.001 for x in tp]
    wrong_2 = [abs(x - 0.0) < 0.001 for x in fp]
    wrong_3 = [abs(x - 0.0) < 0.001 for x in fn]
    print(wrong_1.count(False))
    print(wrong_2.count(False))
    print(wrong_3.count(False))
    print([i for i, a in enumerate(wrong_1) if a == False])
    print([i for i, a in enumerate(wrong_2) if a == False])
    print([i for i, a in enumerate(wrong_3) if a == False])


if __name__ == '__main__':
    path = './test/test_0707_1w.txt'
    eval_signals(path)



