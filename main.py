from Framework import postprocessing_adversarial

experiment = input('Please insert number of experiment to run:')



if experiment == 1:

    class pp_ad(postprocessing_adversarial):
        model_name = 'standard_architectur'

        # learning rate for Adams
        learning_rate = 0.001
        # learning rate adversarial
        learning_rate_adv = 0.001
        # weight adv net
        trans_loss_weight = 1
        # The batch size
        batch_size = 64
        # weight of soft relaxation regulariser adversarial net
        lmb = 10
        # default_adv_steps
        def_adv_steps = 7
        # default_gen_steps
        def_gen_steps = 2

        # noise level
        noise_level = 0.1

    at = postprocessing_adversarial()

    exp = input('Experiment type: ')
    if exp == 1:
        print(at.find_reg_parameter())

    if exp == 2:
        at.train(300)
