from Framework import postprocessing_adversarial

from networks import improved_binary_classifier

experiment = input('Please insert number of experiment to run:')



if experiment == 1:

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'Theory_Parameter'

        # weight adv net
        trans_loss_weight = 0.02

        # noise level
        noise_level = 0.1

        def get_adversarial_network(self):
            return improved_binary_classifier(size=self.image_size, colors=self.colors)

    at = pp_ad()

    exp = input('Experiment type: ')
    if exp == 1:
        print(at.find_reg_parameter())

    if exp == 2:
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
        experiment_name = 'new_architecture'


        # weight adv net
        trans_loss_weight = 0.0003

        # noise level
        noise_level = 0.1

        def get_adversarial_network(self):
            return improved_binary_classifier(size=self.image_size, colors=self.colors)

    at = pp_ad()

    exp = input('Experiment type: ')
    if exp == 1:
        print(at.find_reg_parameter())

    if exp == 2:
        for k in range(5):
            at.train(500)

