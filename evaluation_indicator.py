#
#
#
import math
import numpy as np
from sklearn import metrics

class EvaluationIndicator(object):
    
    def __init__(self, probs, clf, y_test):
        self.probs = probs
        self.clf = clf
        self.y_test = y_test
        pass

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def get_topN_pred(self, n):
        length = len(self.y_test)
        pred = []
        for prob in self.probs:
            p = []
            best_n = np.argsort(prob)[-n:]
            for b in best_n:
                p.append(self.clf.classes_[b])
            pred.append(p)

        tp = 0.
        fp = 0.
        fn = 0.

        for i in xrange(0, length):
            if any(x in pred[i] for x in self.y_test[i]):
                tp += 1
            else:
                fn += 1
                if pred[i]:
                    fp += 1
        
        print "Top %s pr evaluation. tp:%s, fp:%s, fn:%s." % (n, tp, fp, fn)

        print ("precision: %0.4f" % (tp / (tp + fp)))
        print ("recall: %0.4f" % (tp / (tp + fn)))

    def get_threshold_pr(self, step = 0.01):
        length = len(self.y_test)
        print "Threshold pr."
        for threshold in np.arange(0.4, 0.6, step):
            pred = []
            for prob in self.probs:
                p = []
                best_n = np.argsort(prob)[-1:]
                for index in best_n:
                    if self.sigmoid(prob[index]) > threshold:
                        p.append(self.clf.classes_[index])
                pred.append(p)

            tp = 0.
            fp = 0.
            fn = 0.

            for i in xrange(0, length):
                if any(x in pred[i] for x in self.y_test[i]):
                    tp += 1
                else:
                    fn += 1
                    if pred[i]:
                        fp += 1
            print "threshold  precision  recall"
            print "%0.4f\t%0.4f\t%0.4f" % (threshold, (tp / (tp + fp)), (tp / (tp + fn)))

    def get_auc(self):
        print "AUC number."
        pred = []
        y_true = []
        y_score = []
        for prob in self.probs:
            p = []
            best_n = np.argsort(prob)[-1:]
            for index in best_n:
                p.append(self.clf.classes_[index])
            pred.append(p)

        for i in xrange(0, len(self.y_test)):
            for x in pred[i]:
                if x in self.y_test[i]:
                    y_true.append(1)
                else:
                    y_true.append(0)
                y_score.append(self.sigmoid(self.probs[i][pred[i].index(x)]))
        auc_score = metrics.roc_auc_score(y_true, y_score)
        
        print "AUC: %0.4f" % auc_score

