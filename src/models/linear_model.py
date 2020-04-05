
import torch
from torch import nn



class LinearModel(nn.Module):



    def __init__(self,emb_size,  num_classes):
        super(LinearModel, self).__init__()
        self.l1 == nn.Linear(emb_size, num_classes)



    def forward(self, x):
        return nn.Softmax(self.l1(x))