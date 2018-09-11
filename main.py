from Framework import postprocessing
from Framework import iterative_recon
from Framework import postprocessing_supervised
from Framework import iterative_recon_supervised

from data_pips import LUNA
from data_pips import BSDS

from networks import improved_binary_classifier
from networks import resnet_classifier
from networks import Res_UNet

from forward_models import ct


### Experiments on BSDS with 10% noise
nl = 0.1

# the parameter determines the weight of the adv net compared to l2 loss.
# 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param = [0.015, 0.005, 0]
class Exp1(postprocessing):
    noise_level = nl

# the parameter alpha determines the weight of the adv net compared to l2 loss.
# 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param_sup = [1, 0.1]
class Exp2(postprocessing_supervised):
    noise_level = nl

learning_rates = [0.0002, 0.0001, 0.00005, 0.00005]

for rate in learning_rates:
    pass
    # recon = Exp1(reg_param[1], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()

    # recon = Exp1(reg_param[0], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()

    # recon = Exp1(reg_param[2], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()
    #
    # recon = Exp2(reg_param_sup[0], rate)
    # for k in range(10):
    #     recon.pretrain(500)
    # recon.end()

    # recon = Exp2(reg_param_sup[1], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()

### Experiments on LUNA with 3% noise on measurements
nl = 0.02
class Exp3(postprocessing):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()
    def get_model(self, size):
        return ct(size=size)

class Exp4(postprocessing_supervised):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()
    def get_model(self, size):
        return ct(size=size)

class Exp5(iterative_recon):
    noise_level = nl
    def_adv_steps = 8
    def get_Data_pip(self):
        return LUNA()
    def get_model(self, size):
        return ct(size=size)
# the parameter alpha determines the weight of the adv net compared to l2 loss.
# 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param = [0.7, 0.4]

class Exp5_1(Exp5):
    model_name = 'Iterative_Scheme_Stabilized'
    def_adv_steps = 10
reduced_learning_rates = [0.00015, 0.0001, 0.00007, 0.00003, 0.00003, 0.00002]
# the parameter alpha determines the weight of the adv net compared to l2 loss.
# 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param_stabilized = [0.7, 0.4]

class Exp6(iterative_recon_supervised):
    noise_level = nl
    def get_Data_pip(self):
        return LUNA()
    # the parameter alpha determines the weight of the adv net compared to l2 loss.
    # 0 corresponds to pure adversarial loss, 1 to pure l2 loss
reg_param_sup = [1, 0.3]

### do experiments with learned iterative reconstruction only for a start
# use reduced learning rates for unsupervised learning
for rate in learning_rates:
    recon = Exp6(reg_param_sup[0], rate)
    for k in range(10):
        recon.pretrain(500)
    recon.end()
    #
    # recon = Exp6(reg_param_sup[1], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()

    # recon = Exp5(reg_param[0], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()
    #
    # recon = Exp5(reg_param[1], rate)
    # for k in range(10):
    #     recon.train(500)
    # recon.end()

for rate in reduced_learning_rates:
    recon = Exp5_1(reg_param_stabilized[0], rate)
    for k in range(10):
        recon.train(500)
    recon.end()

    recon = Exp5_1(reg_param_stabilized[1], rate)
    for k in range(10):
        recon.train(500)
    recon.end()


