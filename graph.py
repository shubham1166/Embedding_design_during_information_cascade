import numpy as np
import pandas as pd
from tqdm import tqdm as tx
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import laplacian
import torch
import torchvision
import torch.nn as nn
from torch import optim

class Graph:  # This class will trim the dataset to k mostly participating/active users
    def __init__(self, path_to_digg_votes, path_to_digg_friends, tau, test_split, alpha):
        self.digg_votes = np.array(pd.read_csv(path_to_digg_votes, header=None))
        self.digg_friends = np.array(pd.read_csv(path_to_digg_friends, header=None))
        self.tau = tau
        self.test_split = test_split
        self.alpha = alpha
    
    def top_k_nodes(self, dimension):
        unique_elements, counts_elements = np.unique(self.digg_votes[:,1], return_counts=True)
        dec_ix = np.argsort(counts_elements)[::-1]
        return unique_elements[dec_ix[:dimension]]
    
    def trim(self, dimension):
        self.nodes = self.top_k_nodes(dimension)
        self.num_of_nodes = len(self.nodes)
        self.nodes_to_ix = {}
        self.ix_to_node = {}
        for i in range(self.num_of_nodes):
            self.nodes_to_ix[self.nodes[i]] = i
            self.ix_to_node[i] = self.nodes[i]
        # Let us first calculate the Edge Score matrix
        self.S = torch.zeros((self.num_of_nodes, self.num_of_nodes))
        for i in range(len(self.digg_friends)):
            if self.digg_friends[i][0] == 1:
                if self.digg_friends[i][2] in self.nodes:
                    if self.digg_friends[i][3] in self.nodes:
                        a = self.nodes_to_ix[self.digg_friends[i][2]]
                        b = self.nodes_to_ix[self.digg_friends[i][3]]
                        self.S[a,b] = 1
                        self.S[b,a] = 1
                        
        # Let us now calculate the C cascades
        self.C = []
        c = 1
        temp = []
        for i in tx(range(len(self.digg_votes))):
            if self.digg_votes[i][1] in self.nodes:
                if self.digg_votes[i][2] != c:
                    c+=1
                    if temp != []:
                        self.C.append(np.array(temp))
                    temp = []
#                 temp.append(np.array([self.nodes_to_ix[self.digg_votes[i][1]] , self.digg_votes[i][0], self.digg_votes[i][2]]))
                temp.append(np.array([self.nodes_to_ix[self.digg_votes[i][1]] , self.digg_votes[i][0]]))
        self.C_train, self.C_test, _, _ = train_test_split(self.C, self.C, test_size = self.test_split, random_state=42)
        self.num_of_cascades = len(self.C_train)
        #Now Let us calculate X
        self.X = [torch.zeros((self.num_of_nodes, self.num_of_nodes)) for i in range(self.num_of_cascades)]
        self.P = [torch.ones((self.num_of_nodes, self.num_of_nodes)) for i in range(self.num_of_cascades)]
        for i in tx(range(self.num_of_cascades)):
            for j in range(len(self.C_train[i])):
                for k in range(j+1, len(self.C_train[i])):
                    u = self.C_train[i][j][0]
                    v = self.C_train[i][k][0]
                    tu = self.C_train[i][j][1]
                    tv = self.C_train[i][k][1]
                    self.X[i][u,v] = np.exp(-(tu - tv)/self.tau)
                    if self.X[i][u,v] != 0:
                        self.P[i][u,v] = self.alpha
        # Let us now find the affinity matrix A
        self.A = torch.zeros((self.num_of_nodes, self.num_of_nodes))
        for i in tx(range(self.num_of_nodes)):
            for j in range(i+1, self.num_of_nodes):
                c = self.count_affinity(i,j)
                if c!=0:
                    self.A[i,j] = c
                    self.A[j,i] = c
        
        # Let us now define the La and Ls- Laplacian matrix for cascading affinity matrix A and structural matrix S
        self.La = torch.from_numpy(laplacian(np.array(self.A), normed=False))
        self.Ls = torch.from_numpy(laplacian(np.array(self.S), normed=False))
        
    def count_affinity(self, u, v):
        c = 0
        for i in range(self.num_of_cascades):
            if u in self.C_train[i]:
                if v in self.C_train[i]:
                    c+=1
        return c