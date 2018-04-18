from Framework import postprocessing_adversarial

from networks import improved_binary_classifier
from networks import multiscale_l1_classifier
from networks import resnet_classifier
from networks import Res_UNet

experiment = input('Please insert number of experiment to run:')



if experiment == 1:

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'Usual_classifier'

        # learning rate for Adams
        learning_rate = 0.0002
        # learning rate adversarial
        learning_rate_adv = 0.0002

        # default_adv_steps
        def_adv_steps = 12
        # default_gen_steps
        def_gen_steps = 1

        # weight adv net
        trans_loss_weight = 0.015

        # noise level
        noise_level = 0.1

        def get_adversarial_network(self):
            return improved_binary_classifier(size=self.image_size, colors=self.colors)

    at = pp_ad()

    for k in range(5):
        at.train(500)


if experiment == 2:
    print('Train with adversarial loss only for comparison')

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'adv_loss_only'

        # weight adv net
        trans_loss_weight = 0

        # noise level
        noise_level = 0.1

        def get_adversarial_network(self):
            return improved_binary_classifier(size=self.image_size, colors=self.colors)

    at = pp_ad()

    for k in range(5):
        at.train(500)

if experiment == 3:

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'l1_critc'


        # weight adv net
        trans_loss_weight = 0.015

        # noise level
        noise_level = 0.1

        def get_adversarial_network(self):
            return multiscale_l1_classifier(size=self.image_size, colors=self.colors)

    at = pp_ad()

    exp = input('Experiment type: ')
    if exp == 1:
        print(at.find_reg_parameter())

    if exp == 2:
        for k in range(5):
            at.train(500)

if experiment == 4:
    class pp_ad(postprocessing_adversarial):
        experiment_name = 'res_net'


        # weight adv net
        trans_loss_weight = 0.015

        # noise level
        noise_level = 0.1

        def get_adversarial_network(self):
            return resnet_classifier(size=self.image_size, colors=self.colors)

        def get_network(self, size, colors):
            return Res_UNet(size=size, colors=colors)

    at = pp_ad()
    for k in range(5):
        at.train(500)


