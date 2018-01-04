# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import collections
import numpy as np

#seq length 为num_step-1
def batch_iter(x,y,batch_size):
    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        if end_id-start_id<batch_size:
            break
        yield x[start_id:end_id],y[start_id:end_id]

# file -> id  max_length——numsteps  max_data_row——使用训练数据行数
def _file_to_word_ids(filename, word_to_id, max_length=None, max_data_row=None):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        count =0
        for line in lines:
            word_ids=[]
            line = line.strip()

            words = line.split(" ")
            if max_length and len(words) >= max_length:
                continue
            for word in words:
                if word in word_to_id:
                    word_ids.append(word_to_id[word])
                else:
                    word_ids.append(word_to_id['UNK'])
            word_ids.append(word_to_id['ENDMARKER'])
            if max_length:
                for i in range(max_length - len(words)-1):
                    word_ids.append(word_to_id['PAD'])
            count +=1
            data.append(word_ids)
            if max_data_row and count>max_data_row:
                break
    return data


def raw_data(max_data_row,data_path=None, word_to_id=None, max_length=None):
    train_path = os.path.join(data_path,"c/small_train.txt")
    val_path=os.path.join(data_path,'c/val.txt')
    test_path = os.path.join(data_path, "c/test.txt")

    train_data=np.asarray(_file_to_word_ids(train_path, word_to_id, max_length, max_data_row=max_data_row))
    val_data=np.asarray(_file_to_word_ids(val_path,word_to_id,max_length))
    test_data= np.asarray(_file_to_word_ids(test_path, word_to_id, max_length))

    vocabulary_size = len(word_to_id)
    end_id = word_to_id['ENDMARKER']
    left_id = word_to_id['{']
    right_id = word_to_id['}']
    PAD_ID = word_to_id['PAD']
    return train_data,val_data,test_data,vocabulary_size, end_id, left_id, right_id, PAD_ID

def get_pair(data):
    labels=data[:,1:]
    inputs=data[:,:-1]
    return inputs,labels

def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace("\r\n", " ENDMARKER ").split(' ')

def _build_vocab(filename,vocab_size):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, values = list(zip(*count_pairs))
    words = words[0:vocab_size-2]
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['UNK'] = len(word_to_id)
    word_to_id['PAD'] = len(word_to_id)
    return word_to_id


def reverseDic(curDic):
    newmaplist = {}
    for key, value in curDic.items():
        newmaplist[value] = key
    return newmaplist


