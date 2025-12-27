from ptflops import get_model_complexity_info
from net.net import *
import torch
import numpy as np
import time
# from own.mamba import *


from own.LSNet import LSNet
from own.Ours import *

# model = TNet().cuda().eval()
# model = JNet().cuda().eval()
# model = TBNet().cuda().eval()
model = net(ZZ).cuda().eval()
# model = Juan(3,64).cuda().eval()
# model = Backbone().cuda().eval()


H, W = 256, 256
flops_t, params_t = get_model_complexity_info(model, (3, H, W), as_strings=True, print_per_layer_stat=True)
print("Network :FIVE_APLUS")
print(f"net flops:{flops_t} parameters:{params_t}")
# model = nn.DataParallel(model)
x = torch.ones([1, 3, H, W]).cuda()

result = model(x)
steps = 25
# print(b)
time_avgs = []
with torch.no_grad():
    for step in range(steps):

        torch.cuda.synchronize()
        start = time.time()
        result = model(x)
        torch.cuda.synchronize()
        time_interval = time.time() - start
        if step > 5:
            time_avgs.append(time_interval)
        # print('run time:',time_interval)
print('avg time:', np.mean(time_avgs), 'fps:', (1 / np.mean(time_avgs)), ' size:', H, W)