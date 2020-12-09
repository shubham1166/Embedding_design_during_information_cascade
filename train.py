import numpy as np
import pandas as pd
from tqdm import tqdm as tx
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import laplacian
import torch
import torchvision
import torch.nn as nn
from torch import optim
import argparse
from network import AutoEncoder
from network import Graph_embedding
from graph import Graph

parser = argparse.ArgumentParser(description='Graph Embeddings')

parser.add_argument('--seeds', type=int, default = 50, 
                    help='The seed for randomization')

parser.add_argument('--path_to_digg_votes', type=str, default = './Data/digg_votes1.csv',  
                    help='Path to the data file digg_votes1.csv')

parser.add_argument('--path_to_digg_friends', type=str, default = './Data/digg_friends.csv',  
                    help='Path to the data file digg_friends.csv')

parser.add_argument('--alpha', type=float, default = 0.1,  
                    help='Value of alpha in loss function')

parser.add_argument('--beta', type=float, default = 0.9,  
                    help='Value of beta in loss function')

parser.add_argument('--gamma', type=float, default = 0.002,  
                    help='Value of gamma in loss function')

parser.add_argument('--batch_size', type=int, default = 10,  
                    help='Batch size')

parser.add_argument('--path_to_save', type=str, default = './',  
                    help='Path to save the Embeddings')

parser.add_argument('--n_epoch', type=int, default = 100000,  
                    help='Num of epochs')

parser.add_argument('--trim_dimension', type=int, default = 500,  
                    help='No of top nodes we want to take from the whole dataset')

parser.add_argument('--tau', type=int, default = 100000,
                    help='Value of beta in loss function')

parser.add_argument('--test_split', type=float, default = 0.2,
                    help='percentage of test set')

parser.add_argument('--embedding_size', type=int, default = 50,
                    help='Value of beta in loss function')

parser.add_argument('--rho', type=int, default = 2,
                    help='Value of rho used in P matrix')


parser.add_argument('--lr', type=float, default = 0.005,  
                    help="learning rate for the optimizer")

parser.add_argument('--device', type=str, default = 'cpu',  
                    help="device like 'cpu' or 'gpu'")


args = parser.parse_args()


path_to_digg_votes = args.path_to_digg_votes
path_to_digg_friends = args.path_to_digg_friends
batch_size = args.batch_size
n_epoch = args.n_epoch
path_to_save = args.path_to_save
if path_to_save[-1] != '/':
    path_to_save += '/'
trim_dimension = args.trim_dimension
tau = args.tau
test_split = args.test_split
embedding_size = args.embedding_size
rho = args.rho
alpha = args.alpha
beta = args.beta
gamma = args.gamma
device = args.device
lr = args.lr


# Initializing the Graph and the autoencoders
print('Pre-processing the graph dataset') 
G  = Graph(path_to_digg_votes, path_to_digg_friends, tau, test_split, rho)
G.trim(trim_dimension)
device = torch.device(device)
input_size = G.X[0].shape[0]
AE = [AutoEncoder(input_size = input_size).to(device) for i in range(G.num_of_cascades)] #Set of all the autoencoders
Embedding_net = Graph_embedding(embedding_size = embedding_size, output_size = int(input_size/8)).to(device)

#optimizer
params = list(Embedding_net.parameters())
for i in range(G.num_of_cascades):
    params += list(AE[i].parameters())
optimizer = optim.Adam(params, lr=lr)



# Training the model
no_of_iterations = int(G.X[0].shape[0]/batch_size)
for epoch in tx(range(n_epoch)):
    c = 0
    X_cap = []
    for i in range(no_of_iterations):
        if i+1 == no_of_iterations:
            for cascade in range(G.num_of_cascades):
                Input = G.X[cascade][c:].to(device)
                Input = Input.float()
                encoded, decoded = AE[cascade](Input)
                if i == 0:
                    X_cap.append(decoded)
                else:
                    X_cap[cascade] = torch.cat((X_cap[cascade], decoded), axis = 0)
                if cascade == 0:
                    temp = encoded
                else:
                    temp = temp + encoded
            embedding = Embedding_net(temp)
            c += batch_size
        else:
            for cascade in range(G.num_of_cascades):
                Input = G.X[cascade][c:c+batch_size].to(device)
                Input = Input.float()
                encoded, decoded = AE[cascade](Input)
                if i == 0:
                    X_cap.append(decoded)
                else:
                    X_cap[cascade] = torch.cat((X_cap[cascade], decoded), axis = 0)
                if cascade == 0:
                    temp = encoded
                else:
                    temp = temp + encoded
            embedding = Embedding_net(temp)
            c += batch_size
        
        if i == 0:
            Z = embedding
        else:
            Z = torch.cat((Z, embedding), axis = 0)
    
    loss1 = np.sum([torch.norm(torch.mul((G.X[i].to(device) - X_cap[i]), G.P[i].to(device))) for i in range(G.num_of_cascades)])
    loss2 = 2*torch.trace(torch.matmul(torch.matmul(Z.T,G.La.to(device)),Z))
    loss3 = 2*torch.trace(torch.matmul(torch.matmul(Z.T,G.Ls.to(device)),Z))
    l2_reg = torch.tensor(0.).to(device)
    for i in range(G.num_of_cascades):
        for param in AE[i].parameters():
            l2_reg = l2_reg + torch.norm(param)
    for param in Embedding_net.parameters():
        l2_reg = l2_reg + torch.norm(param)
    loss = loss1 + alpha*loss2 + beta*loss3 + gamma*l2_reg
    # Backpropagation and then optimization
    optimizer.zero_grad()#Initially setting the gradient values to zero so backward() can find the gradient
    loss.backward()#backpropagate and then optimize
    optimizer.step()
    print('Epoch',epoch ,'Loss = ', loss.item())
    np.save(path_to_save + 'Z_'+str(embedding_size)+'_Epoch_'+str(epoch)+'_Loss_'+str(loss.item())+'.npy', np.array(Z.detach().cpu()))
    
    
# Defining the accuracy matrices
def prob1(u,v, embd):
    b=(np.linalg.norm(embd[v]-embd[u]))**2
    a=1/(1+np.exp(b))
    return a

def prob_final(u,c, embd):
    sum=0;
    for i in range(math.ceil(0.1*c.shape[0])):
        sum=sum+math.log(1-prob1(u,c[i,0],embd))
    return(sum)

def calc_score(nodes,c, embd):
    r_hat=[]
    noninfected_nodes=list(set(nodes)-set(c[:math.ceil(0.1*c.shape[0]),0]))
    for i in range(len(noninfected_nodes)):
        r_hat.append([noninfected_nodes[i],prob_final(noninfected_nodes[i],c, embd)])
    r_hat.sort(key=lambda x: (x[1]))
    return r_hat
    
def precision(r_hat,r):
    r_hat_set=set(r_hat)
    r_set=set(r)
    return (len(r_hat_set.intersection(r_set))/(len(r_hat_set)+1e-5))

def average_precison(nodes,c,k, embd):
    r_hat=calc_score(nodes,c, embd)
    prcsn=0;
    m=0;
    s=set(c[math.ceil(0.1*c.shape[0]):math.ceil(0.1*c.shape[0])+k,0])
    for j in range(k):
        if(np.array(r_hat)[j,0] in s):
            m+=1;
            prcsn=prcsn+precision(np.array(r_hat)[0:j,0],c[0:,0])
    return (prcsn/m)

def mean_average_precision(nodes,C, embd):
    k=[50, 70, 100,120]
    mean_pre=[]
    for i in range(len(k)):
        mean_prcsn=0
        for j in range(len(C)):
            mean_prcsn += average_precison(nodes,C[j],k[i], embd)
        mean_pre.append(mean_prcsn/len(C))
    return mean_pre
        

nodes = [i for i in range(len(G.S))]

print('MAP at 50, 70, 100,120 is', mean_average_precision(nodes,G.C_test, np.array(Z.detach().cpu())))
