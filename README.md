# Embedding design during information cascade
This is a course project(CS 768) by Shubham Sharma(18i190002) and Pintu Kumar(194193002) under the guidance of Prof. Abir De (IIT Bombay).

The code has an experiment folder that has ipynb notebooks for training. The code can be used directly from train.py. All theoritical explainations are there in the report itself. The code has the following parameters, with their default values as 

*    path_to_digg_votes = './Data/digg_votes1.csv'
*    path_to_digg_friends = './Data/digg_friends.csv'
*    batch_size = 10
*    n_epoch = 100000
*    path_to_save = '/'
*    path_to_load = None
*    phase = 'train'
*    trim_dimension = 500
*    tau = 100000 # By observations
*    test_split = 0.2
*    embedding_size = 100
*    rho = 2
*    lr = 0.005
*    alpha =  0.2
*    beta = 0.8
*    gamma = 0.002
*    device = 'cuda:2'


The code can be run directly as:

python train.py

or if you want to change any value, then 

python train.py --lr 0.2 --test_split 0.2


