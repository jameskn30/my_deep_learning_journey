# import torch
import torch.nn as nn
# from torch.nn.init import xavier_uniform_

class Classifier(nn.Module):
    def __init__(self, num_hiddens):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(num_hiddens, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_score = self.sigmoid(h) * mask_cls.float()
        return sent_score