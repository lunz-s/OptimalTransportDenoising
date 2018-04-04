from Framework import postprocessing_adversarial

experiment = input('Please insert number of experiment to run:')

if experiment == 1:
    at = postprocessing_adversarial()
    at.train(300)
