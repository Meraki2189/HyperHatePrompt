import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc


def eval_MMHS150K(results, truths, exclude_zero=True):
    test_preds = results.view(-1, 2).cpu().detach().numpy()
    # test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    # test_truth = truths.view(-1, 2).cpu().detach().numpy()
    # print("test_preds ", np.array(test_preds))
    test_preds_i = np.argmax(test_preds, axis=1)  # 选最大
    # threshold = 0.75
    # test_preds_i = (test_preds > threshold).astype(int)
    test_truth_i = test_truth
    test_scores = test_preds[:, 1]
    # test_scores = test_preds
    test_scores = np.array(test_scores, dtype='float').tolist()
    # test_truth_i = np.argmax(test_truth, axis=1)
    test_preds_i = np.array(test_preds_i, dtype='int').tolist()
    test_truth_i= np.array(test_truth_i, dtype='int').tolist()
    # np.set_printoptions(threshold=None)
    print("test_preds_i ",np.array(test_preds_i))
    # np.set_printoptions(threshold=None)
    print("test_truth_i ",np.array(test_truth_i))
    f1 = f1_score(test_truth_i, test_preds_i,average='macro')
    acc = accuracy_score(test_truth_i, test_preds_i)
    pre = precision_score(test_truth_i, test_preds_i,average='macro')
    rec = recall_score(test_truth_i, test_preds_i,average='macro')
    roc_auc = roc_auc_score(test_truth_i, test_scores)
    # fpr, tpr, thresholds = roc_curve(test_truth_i, test_preds_i)
    # roc_auc = auc(fpr, tpr)
    return f1, acc, pre, rec, roc_auc, test_scores

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))