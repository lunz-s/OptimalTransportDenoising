from Framework import postprocessing_adversarial

experiment = input('Please insert number of experiment to run:')



if experiment == 1:

    class pp_ad(postprocessing_adversarial):
        experiment_name = 'Parameter_.2'

        # learning rate for Adams
        learning_rate = 0.001
        # learning rate adversarial
        learning_rate_adv = 0.0005
        # weight adv net
        trans_loss_weight = 0.2
        # The batch size
        batch_size = 32
        # weight of soft relaxation regulariser adversarial net
        lmb = 10
        # default_adv_steps
        def_adv_steps = 7
        # default_gen_steps
        def_gen_steps = 2

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

        # learning rate for Adams
        learning_rate = 0.001
        # learning rate adversarial
        learning_rate_adv = 0.0005
        # weight adv net
        trans_loss_weight = 0
        # The batch size
        batch_size = 32
        # weight of soft relaxation regulariser adversarial net
        lmb = 10
        # default_adv_steps
        def_adv_steps = 7
        # default_gen_steps
        def_gen_steps = 2

        # noise level
        noise_level = 0.1

    at = pp_ad()

    at.train(300)


