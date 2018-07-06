import time

import network
import utils


def create_nn_pool():

    batch_size = [512]
    layer_num = range(1, 4)
    layer_sizes =  [128, 256, 512, 1024, 2048]
    activation = 'relu'

    experiment = []

    for b in batch_size:
        for l in layer_num:
            l_array = tuple()
            for n in range(l):
                l_array += (layer_sizes,)
            layers = utils.cartesian(l_array)
            for nn in layers:
                experiment.append({"batch_size": batch_size,
                                   "nn": [(n, activation) for n in nn]})
    return experiment



if __name__ == '__main__':

    # Create pool of network architectures and browse
    results = []
    exp_params = create_nn_pool()
    for e in exp_params:

        # Create new experiment
        nn = FeedForwardNN(batch_size=e["batch_size"], nn_layers=e["nn"])

        # Start and time experiment
        t_i = time.time()
        l, a = nn.run()
        t = time.time() - t_i

        # Add to the results
        results.append({"params": e, "loss": l, "accuracy": a, "time": t})


    # Save results
