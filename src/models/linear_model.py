
import torch
from torch import nn



class LinearModel(nn.Module):



    def __init__(self, num_classes):
        super(LinearModel, self).__init__()
        self.l1 == nn.Linear(self.stacked_embeddings.embedding_length, num_classes)



    def forward(self, x):
        return nn.Softmax(self.l1(x))