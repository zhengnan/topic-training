#
#
#
import sys
import os
import pickle
import traceback

DEFAULT_DATA_PATH = "training-data"
DEFAULT_ROOT_PATH = "/mnt/nanzheng/"
DEFAULT_WORKER_NUMBER = 10
DEFAULT_VECTOR_DIMENSION = 10000

from get_training_data import GetTrainingData
from one_vs_rest_training import OneVsRestTraining
from evaluation_indicator import EvaluationIndicator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.externals import joblib

os.environ['JOBLIB_TEMP_FOLDER'] = '/mnt/'

def start_trainging_process(data_path, root_path, works, dimension, all_data):
    training_data_processor = GetTrainingData(data_path)
    data, topics, target = training_data_processor.get_title_and_content()
    training_processor = OneVsRestTraining(data = data, topics = topics, target = target, root_path = root_path, all_data = all_data)
    clf, x_test, y_test = training_processor.training_process(jobs_number = works, features_number = dimension)
    return clf, x_test, y_test

def get_evaluation(preds, clf, y_test):
    evaluator = EvaluationIndicator(probs = preds, clf = clf, y_test = y_test)
    evaluator.get_topN_pred(1)
    evaluator.get_topN_pred(2)
    evaluator.get_topN_pred(3)
    evaluator.get_topN_pred(4)
    evaluator.get_topN_pred(5)
    evaluator.get_threshold_pr()
    evaluator.get_auc()

if __name__ == "__main__":
    command_data_path = DEFAULT_DATA_PATH
    command_root_path = DEFAULT_ROOT_PATH
    command_worker_number = DEFAULT_WORKER_NUMBER
    command_vector_dimension = DEFAULT_VECTOR_DIMENSION
    command_all_data = False
    test_data_path = None
    command = sys.argv
    if "-d" in command:
        try:
            index = command.index("-d")
            data_path = command[index+1]
            command_data_path = data_path
        except:
            pass
    if "-r" in command:
        try:
            index = command.index("-r")
            root_path = command[index+1]
            command_root_path = root_path
        except:
            pass
    if "-w" in command:
        try:
            index = command.index("-w")
            works = int(command[index+1])
            command_worker_number = works
        except:
            pass
    if "-v" in command:
        try:
            index = command.index("-v")
            vectors = int(command[index+1])
            command_vector_dimension = command_vector_dimension
        except:
            pass
    if "-a" in command:
        command_all_data = True
        try:
            index = command.index("-a")
            test_data_path = command[index+1]
            if os.path.exists(test_data_path) == False:
                command_all_data = False
                test_data_path = None
                print "Test data path %s is not exists! Use default type, 20% training data will be test!" % command[index + 1]
        except:
            command_all_data = False
            print "Test data path error! Use default type, 20% training data will be test!"
    else:
        pass
    if command_all_data:
        print "All data training model!"
    else:
        print "80% data training model!"
    clf, x_test, y_test = start_trainging_process(data_path = command_data_path, root_path = command_root_path, works = command_worker_number, dimension = command_vector_dimension, all_data = command_all_data)
    if command_all_data:
        test_data_processor = GetTrainingData(test_data_path, 0)
        new_test_x, new_test_y, target = test_data_processor.get_title_and_content()
        y_test_new = []
        for text in new_test_x:
            if text in target:
                y_test_new.append(target[text])
            else:
                pass
        new_test_y = y_test_new
        print "New x_test training data's count is %s." % len(new_test_x)
        print "New y_test's content is %s." % len(new_test_y)
        print "Start load vectorizer!"
        vectorizer = joblib.load(os.path.join(command_root_path, "vectorizer.pkl"))
        print "Load vectorizer succeed!"
        test_x = vectorizer.transform(new_test_x)
        try:
            with open(command_root_path + "x_test_after_vectorizer.obj", "wb") as f1, open(command_root_path + "y_test.obj", "wb") as f2:
                pickle.dump(x_test, f1)
                pickle.dump(new_test_y, f2)
            print("Under all data model! Finish write test data into folder!")
        except:
            traceback.print_exc()
        preds = clf.decision_function(test_x)
        get_evaluation(preds, clf, new_test_y)
    else:
        preds = clf.decision_function(x_test)
        get_evaluation(preds, clf, y_test)

