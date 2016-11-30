#
#
#

import logging
import random
import json
import os
import traceback

ARTICLE_CONTENT_FOLDER_PATH = "/mnt/result-json-100"

class GetTrainingData(object):

    def __init__(self, article_folder_path = ARTICLE_CONTENT_FOLDER_PATH, valid_number = 50):
        self.article_folder_path = article_folder_path
        self.valid_number = valid_number
        self.article_training_data = self.load_data()

    def convert_to_utf8(self, data, ignore_dicts=False):
        # if this is a unicode string, return its string representation
        if isinstance(data, unicode):
            return data.encode('utf-8')
        # if this is a list of values, return list of convert_to_utf8 values
        if isinstance(data, list):
            return [convert_to_utf8(item, ignore_dicts=True) for item in data]
        # if this is a dictionary, return dictionary of convert_to_utf8 keys and values
        # but only if we haven't already convert_to_utf8 it
        if isinstance(data, dict) and not ignore_dicts:
            return {
                convert_to_utf8(key, ignore_dicts=True): convert_to_utf8(value, ignore_dicts=True)
                for key, value in data.iteritems()
            }
        # if it's anything else, return it in its original form
        return data

    def load_data(self):
        valid_topic_count = 0
        article_count = 0
        topic_drop = 0
        result = []
        if os.path.exists(self.article_folder_path):
            with open('topic_frequency.txt', 'w') as w:
                for root, dir_names, file_names in os.walk(self.article_folder_path):
                    for file_name in file_names:
                        with open(os.path.join(root, file_name)) as f:
                            try:
                                j = json.load(f)
                                article_result = []
                                valid_article_count_per_topic = 0
                                for article_item in j["article"]:
                                    article_count += 1
                                    for key in article_item:
                                        if key == "title":
                                            valid_article_count_per_topic += 1
                                            article_data_item = (article_item["title"], article_item["content"], file_name, self.convert_to_utf8(article_item["tag"]))
                                            article_result.append(article_data_item)
                                w.write('{0}\t{1}\t{2}\n'.format(file_name, len(j["article"]), valid_article_count_per_topic))
                                if valid_article_count_per_topic > self.valid_number:
                                    result.extend(article_result)
                                    valid_topic_count += 1
                                else:
                                    topic_drop += 1
                            except:
                                traceback.print_exc()
            print("Drop Topic count %d Total article count %d valid topic count %d" % (topic_drop, article_count, valid_topic_count))
        else:
            print "Can't find training data!"
        return result

    def get_title(self):
        titles = []
        topics = []
        target = {}
        for article_item in self.article_training_data:
            titles.append(article_item[0])
            topics.append(article_item[2])
            if article_item[0] not in target:
                target[article_item[0]] = set()
            target[article_item[0]].add(article_item[2])
            target[article_item[0]].add(article_item[3])
        combined = list(zip(titles, topics))
        random.shuffle(combined, lambda: 0)
        titles[:], topics[:] = zip(*combined)
        return titles, topics, target

    def get_title_and_first_paragraph(self):
        text = []
        topics = []
        target = {}
        for article_item in self.article_training_data:
            title = article_item[0]
            paragraph = article_item[1].split('\n')
            first_paragraph = paragraph[0] + " " + paragraph[1]
            temp_text = title + " " + first_paragraph
            text.append(temp_text)
            topics.append(article_item[2])
            if temp_text not in target:
                target[temp_text] = set()
            target[temp_text].add(article_item[2])
            target[temp_text].add(article_item[3])
        combined = list(zip(text, topics))
        random.shuffle(combined, lambda: 0)
        text[:], topics[:] = zip(*combined)
        return text, topics, target

    def get_title_and_content(self):
        text = []
        topics = []
        target = {}
        for article_item in self.article_training_data:
            temp_text = article_item[0]*5 + " " + article_item[1]
            text.append(temp_text)
            topics.append(article_item[2])
            if temp_text not in target:
                target[temp_text] = set()
            target[temp_text].add(article_item[2])
            target[temp_text].add(article_item[3])
        combined = list(zip(text, topics))
        random.shuffle(combined, lambda: 0)
        text[:], topics[:] = zip(*combined)
        return text, topics, target

    def print_article(self):
        print self.article_training_data

if __name__ == "__main__":
    test = GetTrainingData("../training-data")
    #test.get_title()
    a, b, c= test.get_title_and_first_paragraph()
    #test.get_title_and_content()
    print len(c)

