import torch
import torchvision
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(input_size/10)),
            nn.Sigmoid(),
            nn.Linear(int(input_size/10), int(input_size/100)),
            nn.Sigmoid(),
            nn.Linear(int(input_size/100), int(input_size/50)),  
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(input_size/50), int(input_size/100)),
            nn.Sigmoid(),
            nn.Linear(int(input_size/100), int(input_size/10)),
            nn.Sigmoid(),
            nn.Linear(int(input_size/10), input_size),
            nn.Sigmoid(),  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    
class Graph_embedding(nn.Module):
    def __init__(self, embedding_size, output_size):
        super(AutoEncoder, self).__init__()

        self.embedding = nn.Linear(output_size, embedding_size)

    def forward(self, x):
        embd = self.embedding(x)
        return embd
    