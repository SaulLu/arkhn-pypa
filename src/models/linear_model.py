
import torch
from torch import nn



class LinearModel(nn.Module):



    def __init__(self, num_classes):
        super(LinearModel, self).__init__()
        self.init_emb()
        self.l1 == nn.Linear(self.stacked_embeddings.embedding_length, num_classes)

    def init_emb(self):
        # init standard GloVe embedding
        glove_embedding = WordEmbeddings('glove')

        # init Flair forward and backwards embeddings
        flair_embedding_forward = FlairEmbeddings('news-forward')
        flair_embedding_backward = FlairEmbeddings('news-backward')
        # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
        self.stacked_embeddings = StackedEmbeddings([
            glove_embedding,
            flair_embedding_forward,
            flair_embedding_backward,
        ])





    def embed_sent(self, sent: str):
        s = Sentence(sent)
        self.stacked_embeddings.embed(s)
        return torch.cat([tok.embedding for tok in s.tokens])

    def forward(self, x):
        return nn.Softmax(self.l1(x))