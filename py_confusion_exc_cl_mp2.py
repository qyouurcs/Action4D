import sys
import itertools
import os
import pdb
import numpy as np
import glob
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print ('Usage: {0} <img_1> <img_2> ...  <pred_fn1> <pred_fn2> ...'.format(sys.argv[0]))
        sys.exit()

    actions = [
        "None",
        "Drink",
        "Clapping",
        "Reading",
        "Phone call",
        "Interacting phone",
        "Bend",
        "Squad",
        "Wave",
        "Sitting",
        "Pointing to sth",
        "Lift/hold box",
        "Open drawer",
        "Pull/Push sth",
        "Eat from a plate",
        "Yarning /Stretch",
        "Kick"]

    p = []
    gt = []

    if len(sys.argv) % 2 == 0:
        print ('Error image dir and pred fn shoudl be equal.')
        sys.exit()
    
    half = (len(sys.argv)  - 1) // 2

    for ii in range(half):
        d = sys.argv[ii + 1]
        pred_fn = sys.argv[1 + ii + half]
        pred = np.loadtxt(pred_fn)
        print ('pred_fn:',pred_fn)
        print ('image_dir:', d)
        dict_gts = {}
        surfix = ''

        if os.path.isfile(os.path.join(d, 'result.txt')):
            num_lines = 0
            with open(os.path.join(d, 'result.txt')) as fid:
                for aline in fid:
                    num_lines += 1
                    parts = aline.strip().split()
                    if len(parts) == 1:
                        lbl = 0
                    else:
                        lbl = int(parts[1]) + 1
                    dict_gts[num_lines] = lbl
        else:
            surfix = glob.glob(os.path.join(d, '*.txt'))
            surfix = os.path.basename(surfix[0])
            surfix = '_'.join(surfix.split('_')[1:])


        for i in range(pred.shape[0]):

            idx = int(pred[i][0])
            lbl = pred[i][1]

            gt_fn = os.path.join(d, '{}_' + surfix).format(idx)

            if len(dict_gts) > 0:
                gt_lbl = dict_gts[idx]
            elif os.path.isfile(gt_fn):
                with open(gt_fn) as fid:
                    for aline in fid:
                        parts = aline.strip().split()
                        gt_lbl = int(parts[0])
                        if gt_lbl == 12:
                            gt_lbl = 9
                        if gt_lbl >= 13:
                            gt_lbl -= 1
            
            else:
                gt_lbl = int(0)
            p.append(int(lbl))
            gt.append(gt_lbl)
            p_ = np.asarray(p)
            gt_ = np.asarray(gt)
            if i % 100 == 0:
                print (i, (p_ == gt_).sum() * 1.0 / gt_.size)

    p = np.asarray(p)
    gt = np.asarray(gt)
   
    print ('Accuracy', (p == gt).sum() * 1.0 / gt.size)
    # Now smoothing.  
    gap = 3
    for j in range(len(p)):
        if p[j] == gt[j]:
            continue
        for i in range(1,gap+1): 
            if j - i >= 0 and gt[j-i] == p[j]:
                p[j] = gt[j]
                break

            if j + i < len(p) and gt[j+i] == p[j]:
                p[j] = gt[j]
                break
    print ('Now smoothing.')
    p = np.asarray(p)
    gt = np.asarray(gt)
    cnf = confusion_matrix(gt, p)
    plt.figure()
    plot_confusion_matrix(cnf, classes = actions, normalize = True)
    print ('Accuracy', (p == gt).sum() * 1.0 / gt.size)
    plt.show()
