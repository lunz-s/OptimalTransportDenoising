from Framework import postprocessing
from Framework import iterative_recon
from Framework import postprocessing_supervised
from Framework import iterative_recon_supervised

from data_pips import LUNA
from data_pips import BSDS

from networks import improved_binary_classifier
from networks import multiscale_l1_classifier
from networks import resnet_classifier
from networks import Res_UNet


### Experiments on BSDS with 10% noise
nl = 0.1

reg_param = [0.015]
class Exp1(postprocessing):
    noise_level = nl

reg_param_sup = []
class Exp2(postprocessing_supervised):
    noise_level = nl

# find suitable regularization levels
recon = Exp1(0.015)
print(recon.find_reg_parameter())
print(recon.find_reg_parameter_supervised())


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
