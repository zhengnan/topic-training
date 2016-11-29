#
#
#
import numpy as np

class EvaluationIndicator(object):
    
    def __init__(self, probs, clf, y_test):
        self.probs = probs
        self.clf = clf
        self.y_test = y_test
        pass

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

    def get_threshold_pr(self, threshold):
        pass

    def get_auc(self):
        length = len(self.y_test)
        pred = []
        for prob in self.probs:
            p = []
            for index in xrange(0, len(prob)):
                p.append((self.clf.classes_[index], prob[index]))
            pred.append(p)
        t_number = 0.
        f_number = 0.
        p_number = 0.
        t_list = []
        f_list = []
        for i in xrange(0, length):
            for item in pred[i]:
                if item[0] in self.y_test[i]:
                    t_number += 1
                    t_list.append(item[1])
                else:
                    f_number += 1
                    f_list.append(item[1])

        for x in t_list:
            for y in f_list:
                if x > y:
                    p_number += 1

        print "t_number:%s, f_number:%s, p_number:%s" % (t_number, f_number, p_number)
        print "AUC: %0.4f" % (p_number / (t_number * f_number))

