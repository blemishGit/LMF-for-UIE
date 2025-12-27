import torch
from net.ITA import JNet, TNet, GNet, TBNet,UWnet
from net.mamba import JNet_mamba
########################################################################################################################
from own.LSNet import LSNet
from own.USLN import USLN
# from own.LE import *
# from own.FiveA import FIVE_APLUSNet
from own.FiveAOurs import FIVE_APLUSNet
from own.LEOurs import *
# from own.MambaIR import *
# from own.mamba import Backbone
from own.Ours import LNet,LWTNet,Backbone

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.Jnet = FIVE_APLUSNet()
        self.Jnet = Backbone()
        # self.Jnet = VSSConvIntegration(hidden_dim=3, drop_path=0.1, attn_drop_rate=0.1, d_state=16, expand=2.0, is_light_sr=False)
        # self.Jnet = UWnet()
        # self.Jnet = JNet()
        # self.Jnet = JNet_mamba()
        # self.image_net = Mynet()
        # self.image_net = Decoder(device='cuda')
        # self.image_net = Unet()
        self.Tnet = LWTNet()
        # self.Tnet = LNet()
        self.TBnet = LNet()

    def forward(self, data):
        x_j = self.Jnet(data)
        x_t = self.Tnet(data)
        x_tb = self.TBnet(data)
        # X_g = self.Gnet(data)
        # x_a = self.A_net(data)
        return x_j, x_t, x_tb
        # return x_j



