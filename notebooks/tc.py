
import sys
sys.path.append('/home/gurbain/hyq_ml')

import network
import utils

# Readout layer and backprop algo
nn_layers       = [[('osc',)],
                   [('relu', 40)],
                   [('relu', 40)],
                   [('relu', 40)]
                  ]

batch_size      = 2048
max_epochs      = 200
stop_patience   = 150
regularization  = 0.001
metric          = "mae"
optimizer       = "adam"

# ESN
n_res           = 40
n_read          = 30
damping         = 0.1
sparsity        = 0.2
spectral_radius = 0.95
noise           = 0.001
use_real_fb     = False
in_mask         = [True, True]  # None  # No output is injected in the ESN
out_mask        = [False] * 24  # None  # All readouts ouputs are fed back in the ESN

# Other
data_file       = "/home/gurbain/hyq_ml/data/sims/tc.pkl"
save_folder     = "/home/gurbain/hyq_ml/data/nn_learning/" + utils.timestamp()
verbose         = 1
utils.mkdir(save_folder)

nn = network.NN(nn_layers=nn_layers,
                optim=optimizer,
                metric=metric,
                batch_size=batch_size,
                max_epochs=max_epochs,
                regularization=regularization,
                esn_n_res=n_res,
                esn_n_read=n_read,
                esn_in_mask=in_mask,
                esn_out_mask=out_mask,
                esn_real_fb=use_real_fb,
                esn_spec_rad=spectral_radius,
                esn_damping=damping,
                esn_sparsity=sparsity,
                esn_noise=noise,
                data_file=data_file,
                save_folder=save_folder,
                checkpoint=False,
                verbose=verbose,
                random_state=12)


loss, acc = nn.train(evaluate=True, plot_train_states=True, plot_train=True, 
                     plot_hist=True, plot_test_states=True, plot_test=True, win=5000)
