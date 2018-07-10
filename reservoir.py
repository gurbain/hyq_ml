import matplotlib.pyplot as plt
import numpy as np

## MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


class Neuron(object):

    def tran_fct(self, xStates):

        x = xStates.copy()
        for idx, itm in enumerate(x):
            if itm <= 0 :
                x[idx] = 0
            else :
                x[idx] = 0.786*itm/(0.110+itm)-0.014

        return x


class ReservoirNet(Neuron):
    """
    This class implements constant and methods of a artificial neural network
    using Reservoir Computing architecture, with the goal of comparing with a network of spiking populations.
    """

    def __init__(self, n_in=0, n_fb=0, n_res=1000, spec_rad=0.95,
                 leakage=0.1, scale_bias=0.5, scale_fb=5.0, verbose=3,
                 neg_w=True, seed=None, keep_hist=True):

        Neuron.__init__(self)

        self.scale_bias = scale_bias
        self.scale_fb = scale_fb
        self.spec_rad = spec_rad
        self.leakage = leakage
        self.seed = seed
        self.neg_w = neg_w
        self.verbose = verbose
        self.keep_hist = keep_hist

        self.n_in = n_in
        self.n_fb = n_fb
        self.n_res = n_res

    def start(self):

        self.printv("\n\n ===== Initializing Reservoir =====")

        # Get random generator
        if self.seed == None:
            self.rng = np.random.RandomState(np.random.randint(0,99999))
        else:
            self.rng = np.random.RandomState(self.seed)

        # Initialize input weights
        self.w_in = self.rng.randn(self.n_res, self.n_in)
        if not(self.neg_w):
            self.w_in = abs(self.w_in)

        # Initialise bias
        self.w_bias = self.rng.randn(self.n_res, 1) * self.scale_bias

        # Initialize reservoir random weights given spectral radius
        self.w_res = self.get_rand_mat()

        # Initialize reservoir feedback (not used)
        self.w_fb = self.rng.randn(self.n_res, self.n_fb) * self.scale_fb
        if not(self.neg_w):
            self.w_fb = abs(self.w_fb)

        # Create connection probability
        self.p_connect_res = self.create_conn_mat()
        self.p_connect_fb = 0.1
        self.p_connect_in = 1.0

        # Create arrays of zeros for the initial states
        self.x = np.zeros((self.n_res, 1))
        self.u = np.zeros(self.n_in)

        # Set iteration number to 0
        self.it = 0

    def get_coord(self, ind, x_d, y_d):

        if not (isinstance(ind, (int, long)) & isinstance(x_d, (int, long)) & isinstance(y_d, (int, long))):
            raise Exception('population index, x dimension and y dimension must be integer types')
        z_d = x_d * y_d

        z = ind / z_d
        y = (ind - z * z_d) / x_d
        x = ind - z * z_d - y * x_d

        return x, y, z

    def get_prob(self, id_0, id_1, x_d, y_d, c=0.3, lamb=1.0):

        if id_0 == id_1:
            prob = 0.
        else:
            x_0, y_0, z_0 = self.get_coord(id_0, x_d, y_d)
            x_1, y_1, z_1 = self.get_coord(id_1, x_d, y_d)
            d = np.sqrt((x_0 - x_1) ** 2 +
                        (y_0 - y_1) ** 2 + (z_0 - z_1) ** 2)  # eucl distance

            prob = c * np.power(np.e, -np.square(d / lamb))

        return prob

    def create_conn_mat(self):

        p_connect = np.empty((self.n_res, self.n_res))

        for fr in range(self.n_res):
            for to in range(self.n_res):
                p_connect[fr, to] = self.get_prob(to, fr, x_d=3, y_d=3)

        return p_connect

    def get_rand_mat(self):

        mat = self.rng.randn(self.n_res, self.n_res)

        if not(self.neg_w):
            mat = abs(mat)

        w, v = np.linalg.eig(mat)
        mat = np.divide(mat, (np.amax(np.absolute(w)) / self.spec_rad))

        return mat

    def step(self, u=None, y=None):

        if self.verbose > 2:
            print "Iteration " + str(self.it + 1)

        self.u = u
        inp = np.mat(np.dot(self.w_in, self.u) * self.p_connect_in).T

        x_temp = np.dot(self.w_res*self.p_connect_res, self.x) + \
                 self.w_bias  + inp

        self.x = (1 - self.leakage) * self.x + \
                 self.leakage * self.tran_fct(x_temp)

        if self.keep_hist:
            if self.it == 0:
                self.x_hist = self.x.T
                self.u_hist = self.u.T
            else:
                self.x_hist = np.vstack((self.x_hist, self.x.T))
                self.u_hist = np.vstack((self.u_hist, self.u.T))

        self.it += 1

        return self.x

    def run(self, u=None):

        # Random initialization of reservoir weights
        r.start()
        num_it = u.shape[0]

        # Process the input step by step
        self.printv("\n\n ===== Updating Reservoir =====\n")
        for i in range(num_it):
            r.step(inputs[i, :])

    def printv(self, txt):

        if self.verbose > 1:
            print(txt)

    def plot_state_hist(self):

        for i in range(self.n_res):
            plt.plot(self.x_hist[:, i] + i)

        plt.show()

if __name__ == "__main__":

    # Create an impulse array
    inputs = np.zeros((100, 25))
    inputs[10:20, :] = 1.0
    n_in = inputs.shape[1]

    # Create and run reservoir
    r = ReservoirNet(n_in=n_in)
    r.run(inputs)
    r.plot_state_hist()

    print r.x_hist, r.x_hist.shape
