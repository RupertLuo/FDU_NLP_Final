import numpy as np 
import os
import sys
from pprint import pprint

def read_in_data_lists(data_root="../conceptnet_zc/dataset"):
    """
    arrange the training/testing/validation data into paths that can be directly read
    :param data_root:
    :return:
    """
    train_root = os.path.join(data_root, "train")
    train_files = os.listdir(train_root)
    train_paths = list(map(lambda x: os.path.join(train_root, x), [x for x in train_files if 'index' in x]))

    test_root = os.path.join(data_root, "test")
    test_files = os.listdir(test_root)
    test_paths = list(map(lambda x: os.path.join(test_root, x), [x for x in test_files if 'index' in x]))

    dev_root = os.path.join(data_root, "dev")
    dev_files = os.listdir(dev_root)
    dev_paths = list(map(lambda x: os.path.join(dev_root, x), [x for x in dev_files if 'index' in x]))

    return train_paths, test_paths, dev_paths

def loading(namelist, data):
    # bar
    total = len(namelist)
    count_number = 0
    #Totaladjmatrix = np.zeros(shape=(1,500)) #head of numpy, need to be remove in downsteam 
    Indexlist = []
    Referlist = []
    Adjmatrix = []
    for name in namelist:
        count_number += 1
        if count_number % 500 == 0:
            print(str(count_number)+'/'+str(total))
        refer = []
        count = int(name.split('/')[4].split('_')[0]) # the index of the data
        for lines in data:
            match_count = int(lines.split('\t')[0])
            if match_count == count:
                line = lines
                break
        sentence_1 = line.split('\t')[4].split()
        sentence_2 = line.split('\t')[5].split()
        dataname = name.replace('_index', '')
        dataindex = np.load(name).tolist()
        if dataindex[0] != 'None':
            Adjmatrix.append(np.load(dataname).tolist())
        else:
            Adjmatrix.append([])
        refer1 = []
        refer2 = []
        Indexlist.append(dataindex)
        for ref in sentence_1:
            try:
                refer1.append(dataindex.index(ref))
            except:
                pass
        for ref in sentence_2:
            try:
                refer2.append(dataindex.index(ref))      
            except:
                pass  
        refer.append(refer1)
        refer.append(refer2)
        Referlist.append(refer)

    #np.delete(Totaladjmatrix, [0], axis=0)
    return Adjmatrix, Indexlist, Referlist


def running(taskname):
    data_root = '../conceptnet_zc/dataset'
    train_list, test_list, dev_list = read_in_data_lists(data_root)

    with open('../conceptnet_zc/train_new.tsv', 'r', encoding = 'utf-8') as f: #rename
        data = f.readlines()
    if taskname == 'train':
        print('\n\n\n\n*********************\ntrain')
        Totaladjmatrix, Indexlist, Referlist = loading(train_list, data)  #rename
    elif taskname == 'dev':
        print('\n\n\n\n*********************\ndev')
        Totaladjmatrix, Indexlist, Referlist = loading(dev_list, data)  #rename
    elif taskname == 'test':
        print('\n\n\n\n*********************\ntest')
        Totaladjmatrix, Indexlist, Referlist = loading(test_list, data)  #rename
    else:
        raise ValueError
    return Totaladjmatrix, Indexlist, Referlist



# data_root = './dataset'
# train_list, test_list, dev_list = read_in_data_lists(data_root)
#
#
# with open('train_new.tsv', 'r', encoding = 'utf-8') as f: #rename
#     data = f.readlines()
#
#
#
# Totaladjmatrix, Indexlist, Referlist = loading(train_list[0:9], data)  #rename
#
# pprint(Indexlist)