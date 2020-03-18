import configparser
import networkx as nx
import numpy as np

config = configparser.ConfigParser()
config.read("paths.cfg")

class subGraph():
    def __init__(self):
        self.cpnet = None
        self.cpnet_simple = None
        self.concept2id = None
        self.relation2id = None
        self.id2concept = None
        self.id2relation = None
        self.subgraph = None
        self.neibhbornode = None

    def load_resources(self):
        concept2id = {}
        id2concept = {}
        with open("./concept.txt", "r", encoding="utf8") as f:
            for w in f.readlines():
                concept2id[w.strip()] = len(concept2id)
                id2concept[len(id2concept)] = w.strip()
        print("concept2id done")
        id2relation = {}
        relation2id = {}
        with open("./relation.txt", "r", encoding="utf8") as f:
            for w in f.readlines():
                id2relation[len(id2relation)] = w.strip()
                relation2id[w.strip()] = len(relation2id)
        print("relation2id done")

        self.concept2id = concept2id
        self.id2concept = id2concept
        self.id2relation = id2relation
        self.relation2id= relation2id

    def load_cpnet(self):
        print("loading cpnet....")
        self.cpnet = nx.read_gpickle("./cpnet.graph")
        print("Done")

        cpnet_simple = nx.Graph()
        for u, v, data in self.cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        self.cpnet_simple = cpnet_simple

    def generate_points(self, pPoints, nPoints):
        #pPoints store all the feature you got from sentence
        p = []
        n = []
        for point in pPoints:
            if point in self.concept2id:
                p.append(self.concept2id[point])
        for point in nPoints:
            if point in self.concept2id:
                n.append(self.concept2id[point])
        return p, n 

    def get_subgraph(self, pPoints_idx, steps = 0, sortNumber = 500):
        #for one sentence
        '''
        pPoint_idx: id of features
        steps: the distance of nodes we expand
        '''
        self.subgraph = None
        self.neibhbornode = None
        degree = {}
        iter_set = pPoints_idx.union(pPoints_idx) #first sentense
        sub_set = set()  
        store_set = None # store every step for expand
        count = 0
        final_list = []
        for i in iter_set:
            try:
                trynode = self.cpnet_simple[i]
                final_list.append(i)
            except:
                pass
        iter_set = set(final_list)
        while count <= steps:
            count += 1
            temp_set = set()
            for i in iter_set:
                temp_set = temp_set.union(set(self.cpnet_simple[i].keys()))
            sub_set = sub_set.union(temp_set)
            iter_set = temp_set
        subgraph = nx.subgraph(self.cpnet_simple, list(sub_set))
        # Sorted 
        for node in sub_set:
            degree[node] = subgraph.degree(node)

        sub_set = order_dict(degree, sortNumber)
        subgraph = nx.subgraph(self.cpnet_simple, list(sub_set))
        self.subgraph = subgraph
        self.neibhbornode = sub_set

    def subgraph_to_ajmatrix(self):
        indexList = []
        for node in self.neibhbornode:
            # The dictionary of node and feature
            indexList.append(self.id2concept[node])
        adjmatrix = nx.adjacency_matrix(self.subgraph).todense()
        return adjmatrix, indexList
        
def order_dict(dicts, n):
    sortdict = sorted(dicts.items(),key=lambda x:x[1],reverse=True)
    count = 0
    stored = []
    for k,v in sortdict:
        count += 1
        stored.append(k)
        if count >= n:
            break
    return set(stored)
        

if __name__ == "__main__":
    Graph = subGraph()
    Graph.load_cpnet()
    Graph.load_resources()
    dicts = {'a':50,'b': 500, 'c': 1}
    points = {2086, 4519, 1452, 561, 2612, 8789, 13433}
    Graph.get_subgraph(points)
    adjamtrix, indexlist = Graph.subgraph_to_ajmatrix()
    pass 


    # Test if they follow same list 
    FG = nx.Graph()
    FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
    subset = (1,2,3)
    subgraphFG = nx.subgraph(FG, list(subset))
    adjmatrixFG = nx.adjacency_matrix(subgraphFG).todense()
    print(adjmatrixFG)

