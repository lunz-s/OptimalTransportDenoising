from Framework import postprocessing
from Framework import iterative_recon
from Framework import postprocessing_supervised
from Framework import iterative_recon_supervised

from data_pips import LUNA
from data_pips import BSDS

from networks import improved_binary_classifier
from networks import resnet_classifier
from networks import Res_UNet


### Experiments on BSDS with 10% noise
nl = 0.1

# the parameter determines the weight of the adv net compared to l2 loss.
# 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param = [0.015, 0.005, 0.022]

class Exp1(postprocessing):
    noise_level = nl

# the parameter alpha determines the weight of the adv net compared to l2 loss.
# 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param_sup = [1, 0.1, 0.2]
class Exp2(postprocessing_supervised):
    noise_level = nl

recon = Exp1(reg_param[1])
for k in range(10):
    recon.train(500)
recon.end()

recon = Exp1(reg_param[0])
for k in range(10):
    recon.train(500)
recon.end()

recon = Exp2(reg_param_sup[0])
for k in range(10):
    recon.pretrain(500)
recon.end()

recon = Exp2(reg_param_sup[1])
for k in range(10):
    recon.train(500)
recon.end()


### Experiments on LUNA with 3% noise on measurements
nl = 0.02
class Exp3(postprocessing):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()

class Exp4(postprocessing_supervised):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()

class Exp5(iterative_recon):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()

class Exp6(iterative_recon_supervised):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()
