#
#
#
import os
import traceback
import pickle
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from get_training_data import GetTrainingData

os.environ['JOBLIB_TEMP_FOLDER'] = '/mnt/'
base_path = '/mnt/multi_class/title/'
new_file_path = 'new-files'
flag = 1

class OneVsRestTraining(object):
    
    def __init__(self, data, topics, target, root_path = base_path):
        self.data = data
        self.topics = topics
        self.target = target
        self.base_path = root_path
        self.x_train, self.x_test, self.y_train, self.y_test = self.generate_origin_training_data()
    
    def generate_origin_training_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.topics, test_size = 0.2, random_state = 42)
        if flag == 1:
            person = GetTrainingData(article_folder_path = ew_file_path, valid_number = 0)
            new_x_train, new_y_train, new_target = person.get_title_and_content()
            x_train.extend(new_x_train)
            y_train.extend(new_y_train)
            print("Flag is open! Add files in %s" % (new_file_path))
            print("Add new article's number %s." % (len(new_x_train)))
        y_test_new = []
        for text in x_test:
            if text in self.target:
                y_test_new.append(self.target[text])
            else:
                pass
        y_test = y_test_new
        try:
            with open(self.base_path + "y_train.obj", "wb") as f1, open(self.base_path + "y_test.obj", "wb") as f2:
                pickle.dump(y_train, f1)
                pickle.dump(y_test, f2)
            print("Start preprocess!")
            print("Train data count: %d" % len(x_train))
            print("Test data count: %d" % len(x_test))
            print("Train target count: %d" % len(y_train))
            print("Test target count: %d" % len(y_test)
        except:
            traceback.print_exc()
        return x_train, x_test, y_train, y_test

    def clean_and_vectorizer_training_data(self, features_number = 10000):
        vectorizer = HashingVectorizer(stop_words = 'english', non_negative = False, n_features = features_number)
        x_train = vectorizer.transform(self.x_train)
        x_test = vectorizer.transform(self.x_test)
        print("Finish vectorizer!")
        joblib.dump(vectorizer, self.base_path + "vectorizer.pkl")
        try:
            with open(self.base_path + "x_train_after_vectorizer.obj", "wb") as f1, open(self.base_path + "x_test_after_vectorizer.obj", "wb") as f2:
                pickle.dump(x_train, f1)
                pickle.dump(x_test, f2)
            print("Finish write vectorizer data!")
        except:
            traceback.print_exc()
        return x_train, x_test
    
    def training(self, x_train, y_train, jobs_number = 10):
        clf = OneVsRestClassifier(LinearSVC(verbose = True, class_weight = 'balanced'), n_jobs = jobs_number)
        print("Preprocess finished!")
        print("Begin training!")
        clf.fit(x_train, y_train)
        joblib.dump(clf, self.base_path + 'OneVsRestClassifier_LinearSVC')
        clf = joblib.load(self.base_path + 'OneVsRestClassifier_LinearSVC')
        return clf

    def training_process(self, jobs_number = 10, features_number = 10000):
        train_x, test_x = self.clean_and_vectorizer_training_data(features_number)
        clf = self.training(train_x, self.y_train, jobs_number)
        return clf, test_x, self.y_test

if __name__ == "__main__":
    print "This is a joke!"



