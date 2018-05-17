# -*- coding: utf-8 -*-
"""
Created on Tue May 15 23:27:42 2018

@author: batman
"""


import json
import re
import numpy as np

intent_types = [
        'AddToPlaylist','BookRestaurant','GetWeather','RateBook','SearchCreativeWork','SearchScreeningEvent'
                ]

# Function for text preprocessing
def clean_str(string): 
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """ 
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

### Process training data

def process_data():
    train_list = []
    train_label_list = []
    for intent in range(len(intent_types)):
        with open("data/raw_json_data/"+intent_types[intent]+"/train_"+intent_types[intent]+"_full.json",encoding='utf-8') as f:
              data = json.load(f)

        
        sent_list =[]
        label_list =[]
        for i in range(len(data[intent_types[intent]])):
            txt = ''
            for j in range(len(data[intent_types[intent]][i]['data'])):
                txt = txt+data[intent_types[intent]][i]['data'][j]['text']
            txt = clean_str(txt)
            sent_list.append(txt)
            
        for  i in range(len(sent_list)):
            label_list.append(intent_types[intent])
            
        filename = intent_types[intent]+'_train.txt'

        with open("data/processed_data/"+filename, mode="w", encoding='utf-8') as outfile:
            for s in sent_list:
                outfile.write("%s\n" % s)
        train_list.extend(sent_list)
        train_label_list.extend(label_list)
                
    ### Process test data
    test_list = []
    test_label_list = []
    for intent in range(len(intent_types)):
        with open("data/raw_json_data/"+intent_types[intent]+"/validate_"+intent_types[intent]+".json",encoding='utf-8') as f:
              data = json.load(f)

        
        sent_list =[]
        label_list =[]
        for i in range(len(data[intent_types[intent]])):
            txt = ''
            for j in range(len(data[intent_types[intent]][i]['data'])):
                txt = txt+data[intent_types[intent]][i]['data'][j]['text']
            txt = clean_str(txt)
            sent_list.append(txt)
            
        for  i in range(len(sent_list)):
            label_list.append(intent_types[intent])
        filename = intent_types[intent]+'_test.txt'

        with open("data/processed_data/"+filename, mode="w", encoding='utf-8') as outfile:
            for s in sent_list:
                outfile.write("%s\n" % s)
        test_list.extend(sent_list)  
        test_label_list.extend(label_list)  

        # how to save list as array: np.array(myList).dump(open('array.npy', 'wb'))
        # how to load aan array: myArray = np.load(open('array.npy', 'rb'))
        np.array(train_list).dump(open('data/train_text.npy', 'wb'))
        np.array(train_label_list).dump(open('data/train_label.npy', 'wb'))
        np.array(test_list).dump(open('data/test_text.npy', 'wb'))
        np.array(test_label_list).dump(open('data/test_label.npy', 'wb'))





if __name__ == '__main__':
    process_data()