#
#
#
import sys

DEFAULT_DATA_PATH = "training-data"
DEFAULT_ROOT_PATH = "/mnt/nanzheng/"
DEFAULT_WORKER_NUMBER = 10
DEFAULT_VECTOR_DIMENSION = 10000

from get_training_data import GetTrainingData
from one_vs_rest_training import OneVsRestTraining
from evaluation_indicator import EvaluationIndicator

def start_trainging_process(data_path, root_path, works, dimension):
    training_data_processor = GetTrainingData(data_path)
    data, topics, target = training_data_processor.get_title_and_content()
    training_processor = OneVsRestTraining(data = data, topics = topics, target = target, root_path = root_path)
    clf, x_test, y_test = training_processor.training_process(jobs_number = works, features_number = dimension)
    return clf, x_test, y_test

def get_evalauation():
    pass

if __name__ == "__main__":
    command_data_path = DEFAULT_DATA_PATH
    command_root_path = DEFAULT_ROOT_PATH
    command_worker_number = DEFAULT_WORKER_NUMBER
    command_vector_dimension = DEFAULT_VECTOR_DIMENSION
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
            command_vector_dimension = vectors
        except:
            pass
    else:
        pass
    clf, x_test, y_test = start_trainging_process(data_path = command_data_path, root_path = command_root_path, works = command_worker_number, dimension = command_vector_dimension)
    pred = clf.decision_function(x_test)
    evaluator = EvaluationIndicator(probs = pred, clf = clf, y_test = y_test)
    evaluator.get_topN_pred(1)
    evaluator.get_topN_pred(2)
    evaluator.get_topN_pred(3)
    evaluator.get_topN_pred(4)
    evaluator.get_topN_pred(5)
    evaluator.get_auc()

