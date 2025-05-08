import torch

from Ours import LNet,LWTNet,Backbone

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Jnet = Backbone()
        self.Tnet = LWTNet()
        self.TBnet = LNet()

    def forward(self, data):
        x_j = self.Jnet(data)
        x_t = self.Tnet(data)
        x_tb = self.TBnet(data)
        return x_j, x_t, x_tb



