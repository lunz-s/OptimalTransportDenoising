from Framework import postprocessing_adversarial

experiment = input('Please insert number of experiment to run:')



if experiment == 1:

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'Theory_Parameter'

        # weight adv net
        trans_loss_weight = 0.001

        # noise level
        noise_level = 0.1

    at = pp_ad()

    exp = input('Experiment type: ')
    if exp == 1:
        print(at.find_reg_parameter())

    if exp == 2:
        at.train(300)


if experiment == 2:
    print('Train with adversarial loss only for comparison')

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'adv_loss_only'

        # weight adv net
        trans_loss_weight = 0

        # noise level
        noise_level = 0.1

    at = pp_ad()

    at.train(300)

if experiment == 3:

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'High_transp_weight'


        # weight adv net
        trans_loss_weight = 0.2

        # noise level
        noise_level = 0.1

    at = pp_ad()

    exp = input('Experiment type: ')
    if exp == 1:
        print(at.find_reg_parameter())

    if exp == 2:
        at.train(300)

