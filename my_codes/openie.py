from openie import StanfordOpenIE
from utils_task1 import *
def read_sentence_dataset(data_path):
    with open(data_path,'r',encoding='utf8') as f:
        result = []
        for line in f:
            line_list = line.split()
            idx = line_list[0]
            idx2 = line_list[1]
            idx3 = line_list[2]
            sen1 = line_list[3]
            sen2 = line_list[4]
            result.append((idx,idx2,idx3,sen1,sen2))
    return result
def extract_entity(triple):
    entity = []
    for t in triple:
        entity.append(t['subject'])
        entity.append(t['object'])
    entity = list(set(entity))
    return entity
dev = read_sentence_dataset('/root/nlp_final/my_codes/dataset/task1/dev.tsv')
test= read_sentence_dataset('/root/nlp_final/my_codes/dataset/task1/test.tsv')
train= read_sentence_dataset('/root/nlp_final/my_codes/dataset/task1/train.tsv')
with StanfordOpenIE() as client:
    dev_new = open('/root/nlp_final/my_codes/dataset/task1/dev.tsv','w',encoding = 'utf8')
    train_new = open('/root/nlp_final/my_codes/dataset/task1/train.tsv','w',encoding = 'utf8')
    test_new = open('/root/nlp_final/my_codes/dataset/task1/train.tsv','w',encoding = 'utf8')

    for line in dev:
        text1 = line[3]
        text2 = line[4]
        entity1 = " ".join(extract_entity(client.annotate(text1)))
        entity2 = " ".join(extract_entity(client.annotate(text2)))
        dev_new.write(" ".join(line)+"  "+'1'+' '+entity1+" "+"2"+" "+entity2+'\n')
    for line in test:
        text1 = line[3]
        text2 = line[4]
        entity1 = " ".join(extract_entity(client.annotate(text1)))
        entity2 = " ".join(extract_entity(client.annotate(text2)))
        test_new.write(" ".join(line)+"  "+'1'+' '+entity1+" "+"2"+" "+entity2+'\n')
    for line in train:
        text1 = line[3]
        text2 = line[4]
        entity1 = " ".join(extract_entity(client.annotate(text1)))
        entity2 = " ".join(extract_entity(client.annotate(text2)))
        train_new.write(" ".join(line)+"  "+'1'+' '+entity1+" "+"2"+" "+entity2+'\n')
    

