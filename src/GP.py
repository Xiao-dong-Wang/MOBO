import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b
from .util import chol_inv
import traceback
import sys

# A conventional gaussian process class for bayesian optimization
class GP:
    # Initialize GP class
    # train_x shape: (dim, num_train);   train_y shape: (1, num_train) 
    def __init__(self, dataset, bfgs_iter=100, debug=True):
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.normalize()
        self.jitter = 1e-9

    # Normalize y
    def normalize(self):
        self.mean = self.train_y.mean()
        self.std = self.train_y.std() + 0.000001
        self.train_y = (self.train_y - self.mean)/self.std

    # Initialize hyper_parameters
    def get_default_theta(self):
        # sn2, output_scale, length_scale
        theta = np.random.randn(2 + self.dim)
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        theta[0] = np.log(np.std(self.train_y)) # sn2
        return theta

    # Rbf kernel
    def kernel(self, x, xp, theta):
        output_scale = np.exp(theta[1])
        lengthscales = np.exp(theta[2:]) + 0.000001
        diffs = np.expand_dims((x.T/lengthscales).T, 2) - np.expand_dims((xp.T/lengthscales).T, 1)
        return output_scale * np.exp(-0.5*np.sum(diffs**2, axis=0))
    
    def neg_log_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        K = self.kernel(self.train_x, self.train_x, theta) + sn2*np.eye(self.num_train) + self.jitter*np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        logDetK = np.sum(np.log(np.diag(L)))
        alpha = chol_inv(L, self.train_y.T)
        nlz = 0.5*(np.dot(self.train_y, alpha) + self.num_train*np.log(2*np.pi)) + logDetK
        if(np.isnan(nlz)):
            nlz = np.inf

        self.nlz = nlz
        return nlz

    # Minimize the negative log-likelihood
    def train(self):
        theta0 = self.get_default_theta()
        self.loss = np.inf
        self.theta = np.copy(theta0)

        nlz = self.neg_log_likelihood(theta0)

        def loss(theta):
            nlz = self.neg_log_likelihood(theta)
            return nlz

        def callback(theta):
            if self.nlz < self.loss:
                self.loss = self.nlz
                self.theta = np.copy(theta)

        gloss = value_and_grad(loss)

        try:
            fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m = 100, iprint=self.debug, callback=callback)
        except np.linalg.LinAlgError:
            print('GP. Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=callback)
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())

        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('GP. Failed to build GP model')
            sys.exit(1)


        sn2 = np.exp(self.theta[0])
        K = self.kernel(self.train_x, self.train_x, self.theta) + sn2 * np.eye(self.num_train) + self.jitter*np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y.T)
        self.for_diag = np.exp(self.theta[1])
        print('GP. GP model training process finished')

    def predict(self, test_x, is_diag=1):
        sn2 = np.exp(self.theta[0])
        K_star = self.kernel(test_x, self.train_x, self.theta)
        py = np.dot(K_star, self.alpha)
        KvKs = chol_inv(self.L, K_star.T)
        if is_diag:
            ps2 = self.for_diag + sn2 - (K_star * KvKs.T).sum(axis=1)
        else:
            ps2 = sn2 - np.dot(K_star, KvKs) + self.kernel(test_x, test_x, self.theta)
        ps2 = np.abs(ps2)
        py = py * self.std + self.mean
        py = py.reshape(-1)
        ps2 = ps2 * (self.std**2)
        return py, ps2
    
