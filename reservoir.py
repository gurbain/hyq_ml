
class Neuron(object):

    def tran_fct(self, xStates):

        x = xStates.copy()
        for idx,itm in enumerate(x):
            if itm <= 0 :
                x[idx] = 0
            else :
                x[idx] = 0.786*itm/(0.110+itm)-0.014 #fitted IO curve
        return x

        # return np.tanh(xStates)

class ReservoirNet(Neuron):
    """
    This class implements constant and methods of a artificial neural network
    using Reservoir Computing architecture, with the goal of comparing with a network of spiking populations.
    """

    def __init__(self, n_in=0, n_fb=0, n_out=1, n_res=100, spec_rad=1.15, leakage=0.1, scale_bias=0.5, scale_fb=5.0, scale_noise=0.01, scale_fb_noise=0.01,
                 verbose=False,negative_weights=True,fraction_inverted=0.0,seed=None,config=None):
        """
        :param n_in:
        :param n_fb:
        :param n_out:
        :param n_res:
        :param spec_rad:
        :param leakage:
        :param scale_bias:
        :param scale_fb:
        :param scale_noise:
        :param scale_fb_noise:
        :param verbose:
        :param negative_weights:
        :param fraction_inverted:
        :param seed:
        :param config:
        """
        if config != None:
            self.load(config)
        else:
            Neuron.__init__(self)

            self.scale_bias = scale_bias
            self.scale_fb = scale_fb
            self.scale_noise = scale_noise
            self.scale_feedback_noise = scale_fb_noise

            self.leakage = leakage

            self.TRANS_PERC = 0.1

            self.n_in = n_in
            self.n_fb = n_fb
            self.n_out = n_out
            self.n_res = n_res

            if seed == None:
                self.rng=np.random.RandomState(np.random.randint(0,99999))
            else:
                self.rng=np.random.RandomState(seed)

            self.w_out = self.rng.randn(n_out,n_res) # (to,from)
            if not (negative_weights):
                self.w_out = abs(self.w_out)
            self.w_in = self.rng.randn(n_res, n_in)
            if not(negative_weights):
                self.w_in = abs(self.w_in)
            self.w_bias = self.rng.randn(n_res,1) * self.scale_bias
            self.w_bias = 0.0 #mimics resting noise level of approx 50 Hz
            self.w_res = self.get_rand_mat(n_res, spec_rad,negative_weights=negative_weights) # (to,from)

            self.w_fb = self.rng.randn(n_res, n_fb) * self.scale_fb
            # while(min(self.w_fb)<-0.5): #large negative fb weight results in inactive population
            #     self.w_fb[self.w_fb.argmin()] = self.rng.randn() * self.scale_fb

            if not(negative_weights):
                self.w_fb = abs(self.w_fb)

            self.p_connect_res = self.createConnectivityMatrix() # (to,from)
            # p_connect_fb = np.ones((n_res))*0.1 # fixed Pconnect for feedback connections
            # self.p_connect_fb = p_connect_fb.reshape(-1,1)
            # self.p_connect_in = np.ones((n_res,1))
            self.p_connect_fb = 0.1
            self.p_connect_in = 1.0

            self.N_inverted = int(np.round(n_res*fraction_inverted))
            # set initial state
            # self.x = self.rng.randn(n_res, 1)
            self.x = np.zeros((n_res, 1))
            self.u = np.zeros(n_in)
            self.y = np.zeros((n_out, 1))

            self.verbose = verbose

            self.Y = np.array([])
            self.X = np.array([])

    def get_coord(self, ID, xD, yD):

        if not (isinstance(ID, (int, long)) & isinstance(xD, (int, long)) & isinstance(yD, (int, long))):
            raise Exception('population ID, xDimension and yDimension must be integer types')
        zD = xD * yD

        z = ID / zD
        y = (ID - z * zD) / xD
        x = ID - z * zD - y * xD
        return x, y, z

    def get_prob(self, ID0, ID1, xD, yD, C=0.3, lamb=1.0):

        if ID0 == ID1:
            prob = 0.
        else:
            x0, y0, z0 = self.getCoordinates(ID0, xD, yD)
            x1, y1, z1 = self.getCoordinates(ID1, xD, yD)
            d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)  # eucl distance
            prob = C * np.power(np.e, -np.square(d / lamb))
        return prob

    def create_conn_mat(self):
        p_connect = np.empty((self.n_res,self.n_res))
        for fr in range(self.n_res):
            for to in range(self.n_res):
                p_connect[fr,to] = self.getProb(to,fr,xD=3,yD=3) # (to,from)
        return p_connect

    def get_rand_mat(self, dim, spec_rad,negative_weights=True):

        mat = self.rng.randn(dim, dim)
        if not(negative_weights):
            mat = abs(mat)
        w, v = np.linalg.eig(mat)
        mat = np.divide(mat, (np.amax(np.absolute(w)) / spec_rad))

        return mat


    def update_state_quick(self, u=None, y=None):

        t0 = time.time()
        inp = np.dot(self.w_in, u) * self.p_connect_in
        t1 = time.time()

        ## calculate x_new = x_prev*w + bias  + fb + input + noise
        x_temp = np.dot(self.w_res*self.p_connect_res, self.x)# + self.w_bias  + inp + noise+ fb
        t1 = time.time()

        ## calculate new reservoir state with leakage
        self.x = (1 - self.leakage) * self.x + self.leakage * self.tran_fct(x_temp)
        t2 = time.time()

        return

