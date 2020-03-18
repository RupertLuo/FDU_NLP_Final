import scipy as sp 
import numpy as np
import sys
from subgraph import subGraph
with open('dev_new.tsv', 'r', encoding = 'utf-8') as f:
    data = f.readlines()

Graph = subGraph()
Graph.load_cpnet()
Graph.load_resources()
indexLine = [] 
##########
length = len(data)
length50 = length/50
done = 0
Totaladjmatrix = np.zeros(shape=(1,500))
###########
for lines in data:
    line = lines.rstrip('\n')
    sentenceID = line.split('\t')[0]
    count = int(sentenceID)
    sentence_1 = line.split('\t')[4].split()
    sentence_2 = line.split('\t')[5].split()
    points_1, points_2 = Graph.generate_points(sentence_1, sentence_2)
    points = set(points_1 + points_2)
    Graph.get_subgraph(points)
    try:
        adjmatrix, indexlist = Graph.subgraph_to_ajmatrix()
        # if len(indexlist) < 500:
        #    indexlist += ['None' * (500 - len(indexlist))]        
        # indexLine.append(indexlist)
        # Totaladjmatrix = np.concatenate((Totaladjmatrix, adjmatrix), axis=0)
        name = './dataset/dev/' + sentenceID + '.npy'
        indexname = './dataset/dev/' + sentenceID + '_index.npy'
        adjamtrix = adjmatrix.astype(np.bool).astype(np.int8)
        Indexarray = np.array(indexlist,dtype=str)
        np.save(name, adjmatrix)
        np.save(indexname,Indexarray)
    except:
        # indexlist = ['None'] * 500
        # indexLine.append(indexlist)
        # indexLine.append(['None'])
        indexlist = ['None']
        Indexarray = np.array(indexlist,dtype=str)
        indexname = './dataset/dev/' + sentenceID + '_index.npy'
        np.save(indexname,Indexarray)

    if count >= (done+1) * length50:
        done += 1
    sys.stdout.write("\r[%s%s] %d/%d" % ('█' * done, ' ' * (50 - done), count, length))
    sys.stdout.flush()




#name = './data/data.npy'
#np.save(name, Totaladjmatrix)
#Indexarray = np.array(indexLine,dtype=str)
'''
name = './data/Index.npy'
np.save(name, Indexarray)
'''
'''
with open('./data/index.txt', 'w',encoding='utf-8') as f:
    for line in indexLine:
        for index in line: 
            f.write(index) 
            f.write('\t')
        f.write('\n')
'''


