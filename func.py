
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import norm


def get_data(hierar = False, oneHot = False):

    #Gets path to folder
    path = os.getcwd()
    train_path = path + '/train_heart.csv'
    test_path = path + '/test_heart.csv'

    #variable for the categorical data columns in the dataset
    col = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    #print(col)
    # Load Data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if hierar:
        age_train = df_train['sex']
        age_test= df_test['sex']
        age = [age_train, age_test ]
    else: 
        age = []

    if oneHot:
        # One hot encode data: 
        X_train = pd.get_dummies(df_train,columns=col)
        X_test = pd.get_dummies(df_test,columns=col)
        
        #Extract y values:
        y_train = X_train['target'].values.astype("int")
        y_test = X_test['target'].values.astype("int")

        names = list(X_train.drop("target", axis='columns').columns)

        #Extract data 
        X_train = X_train.loc[:, X_train.columns != 'target'].values
        X_test = X_test.loc[:, X_test.columns != 'target'].values

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_train[:,0:5].astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_train[:,0:5] = X_train_temp

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_test[:,0:5].astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_test[:,0:5] = X_train_temp
    else:
        X_train = df_train
        X_test = df_test

        #Extract y values:
        y_train = X_train['target'].values.astype("int")
        y_test = X_test['target'].values.astype("int")
        
        names = list(X_train.drop("target", axis='columns').columns)

        #Extract data 
        X_train = X_train.loc[:, X_train.columns != 'target'].values
        X_test = X_test.loc[:, X_test.columns != 'target'].values

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_train.astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_train = X_train_temp

        #Normalize non one-hot-encoded data (training data)
        X_train_temp = X_test.astype('float')
        X_mean = X_train_temp.mean(axis=0)
        X_std = X_train_temp.std(axis=0)
        X_train_temp = (X_train_temp - X_mean) / X_std
        X_test = X_train_temp
  

    return [X_train,y_train,X_test,y_test,age,names]

sigmoid = lambda x: 1./(1+np.exp(-x))
probit = lambda x: norm.cdf(x)

class BernoulliLikelihood(object):
    """ Implement the Bernoulli likelihood with the sigmoid as inverse link function """
    

    def __init__(self, y):
        # store data & force shape (N, )
        self.y = y.ravel()

    def log_lik(self, f):
        """ Implements log p(y|f) = sum log p(y_n|f_n), where p(y_n|f_n) = Ber(y_n|sigmoid(f_n)). 
            
            Argument:
            f       --       vector of function values, shape (N, )

            Returns
            ll      --       sum of log likelihoods for all N data points, scalar

        """
        ##############################################
        # Your solution goes here
        ##############################################
        
        p = sigmoid(f)
        ll = np.sum(self.y*np.log(p) + (1-self.y)*np.log(1-p))
        
        ##############################################
        # End of solution
        ##############################################

        # check shape and return
        assert ll.shape == (), f"Expected shape for loglik_ is (), but the actual shape was {ll.shape}. Please check implementation"
        return ll
    
    def grad(self, f):
        """ Implements the gradient of log p(y|n) 

            Argument:
            f       --       vector of function values, shape (N, )

            Returns
            g       --       gradient of log p(y|f), i.e. a vector of first order derivatives with shape (N, )
             
        """
        ##############################################
        # Your solution goes here
        ##############################################
        
        g = self.y - sigmoid(f)
        
        ##############################################
        # End of solution
        ##############################################
        # check shape and return
        assert g.shape == (len(f), ), f"Expected shape for g is ({len(f)}, ), but the actual shape was {g.shape}. Please check implementation"
        return g

    def hessian(self, f):
        """ Implements the Hessian of log p(y|n) 

        Argument:
            f       --       vector of function values, shape (N, )

        Returns:
            Lambda  --       Hessian of likelihood, i.e. a diagonal matrix with the second order derivatives on the diagonal, shape (N, N)
        """

        ##############################################
        # Your solution goes here
        ##############################################
        
        p = sigmoid(f)
        Lambda = np.diag(-p*(1-p))    
        
        ##############################################
        # End of solution
        ##############################################

        # check shape and return
        assert Lambda.shape == (len(f), len(f)), f"Expected shape for Lambda is ({len(f)}, {len(f)}), but the actual shape was {Lambda.shape}. Please check implementation"
        return Lambda


class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun, kappa=1., lengthscale=1.0):
        """
            the argument kernel_fun must be a function of three arguments kernel_fun(||tau||, kappa, lengthscale), e.g. 
            squared_exponential = lambda tau, kappa, lengthscale: kappa**2*np.exp(-0.5*tau**2/lengthscale**2)
        """
        self.kernel_fun = kernel_fun
        self.kappa = kappa
        self.lengthscale = lengthscale

    def contruct_kernel(self, X1, X2, kappa=None, lengthscale=None, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun
    
        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix
            kappa           -- magnitude (positive scalar)
            lengthscale     -- characteristic lengthscale (positive scalar)
            jitter          -- non-negative scalar
        
        returns
            K               -- NxM matrix    
        """

        # extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        # prep hyperparameters
        kappa = self.kappa if kappa is None else kappa
        lengthscale = self.lengthscale if lengthscale is None else lengthscale

        # compute all the pairwise distances efficiently
        dists = np.sqrt(np.sum((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2, axis=-1))
        
        # squared exponential covariance function
        K = self.kernel_fun(dists, kappa, lengthscale)
        
        # add jitter to diagonal for numerical stability
        if len(X1) == len(X2) and np.allclose(X1, X2):
            K = K + jitter*np.identity(len(X1))
                
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K

# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(tau, kappa, lengthscale):
    return kappa**2*np.exp(-0.5*tau**2/lengthscale**2)


def compute_err(t, tpred):
    return np.mean(tpred.ravel() != t), np.std(tpred.ravel() != t)/np.sqrt(len(t))



class GaussianProcessClassification(object):

    def __init__(self, X, y, likelihood, kernel, kappa=1., lengthscale=1.,jitter=1e-8):
        """  
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values 
            likelihood       -- likelihood instance
            kernel           -- must be instance of the StationaryIsotropicKernel class
            jitter           -- non-negative scaler
            kappa            -- magnitude (positive scalar)
            lengthscale      -- characteristic lengthscale (positive scalar)
        """
        self.X = X
        self.y = y
        self.N = len(X)
        self.likelihood = likelihood(y)
        self.kernel = kernel
        self.jitter = jitter
        self.set_hyperparameters(kappa, lengthscale)

        # precompute kernel, its Cholesky decomposition and prepare Laplace approx
        self.K = self.kernel.contruct_kernel(self.X, self.X, jitter=self.jitter)
        self.L = np.linalg.cholesky(self.K)
        self.construct_laplace_approximation()

    def set_hyperparameters(self, kappa, lengthscale):
        self.kernel.kappa = kappa
        self.kernel.lengthscale = lengthscale
        
    def log_joint_a(self, a):
        """ computes and returns the log joint distribution log p(y, f), where f = K*a """
        f = self.K@a
        # compute log prior contribution
        const = -self.N/2*np.log(2*np.pi)
        logdet = np.sum(np.log(np.diag(self.L)))
        quad_term =  0.5*np.sum(a*f)
        log_prior = const - logdet - quad_term
        # compute log likelihood contribution
        log_lik = self.likelihood.log_lik(f)
        # return sum
        return log_prior + log_lik
    

    def grad_a(self, a):
        """ computes gradient of log joint distribution, i.e. log p(y, a) = log p(y|a) + log p(a), wrt. a """
        f = self.K@a
        # compute gradient contribution from prior and likelihood
        grad_prior = -f
        grad_lik = self.likelihood.grad(f)@self.K
        # sum and return
        return grad_prior + grad_lik
        
    
    def compute_f_MAP(self):
        # optimize to get f_MAP
        result = minimize(lambda a: -self.log_joint_a(a), jac=lambda a: -self.grad_a(a), x0=np.zeros((self.N)))
        
        if not result.success:
            print(result)
            raise ValueError('Optization failed')
        
        self.a = result.x
        f_MAP = self.K @ result.x
        return f_MAP

    def construct_laplace_approximation(self):

        # f_MAP
        self.m = self.compute_f_MAP()

        # Compute Hessian
        
        Lambda = -self.likelihood.hessian(self.m)

        # straigth-forward implementation of S
        # self.H = -Lambda - np.linalg.inv(self.K)
        # self.S = np.linalg.inv(-self.H)
        # numerically more robust approach for computing S
        Lsqrt = np.sqrt(Lambda)
        B = np.identity(len(self.m)) + Lsqrt@self.K@Lsqrt
        chol_B = np.linalg.cholesky(B)
        e = np.linalg.solve(chol_B, Lsqrt@self.K)
        
        self.S = self.K - e.T@e  

    def predict_f(self, Xstar):
        """ returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- mean vector, shape (P,)
        Sigma            -- covariance matrix, shape (P, P) 
        """
        ##############################################
        # Your solution goes here
        ##############################################
        
        k = self.kernel.contruct_kernel(Xstar, self.X, jitter=self.jitter)
        Kp = self.kernel.contruct_kernel(Xstar, Xstar, jitter=self.jitter)

        h = np.linalg.solve(self.K, k.T)
        mu = k@np.linalg.solve(self.K, self.m)
        Sigma = Kp - h.T@(self.K-self.S)@h
        
        ##############################################
        # End of solution
        ##############################################

        # check dimensions and return
        assert (mu.shape == (len(Xstar),)), f"Expected shape for mu is ({len(Xstar)}), but the actual shape was {mu.shape}. Please check implementation"
        assert Sigma.shape == (len(Xstar), len(Xstar)), f"Expected shape for Sigma is ({len(Xstar)}, {len(Xstar)}), but the actual shape was {Sigma.shape}. Please check implementation"

        return mu, Sigma
    
    def predict_y(self, Xstar):
        """ returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        p               -- vector of post. pred. probabilities, shape (P,)
        """
        ##############################################
        # Your solution goes here
        ##############################################
        
        mu, Sigma = self.predict_f(Xstar)
        p = probit(mu/np.sqrt(8/np.pi + np.diag(Sigma)))
        
        ##############################################
        # End of solution
        ##############################################

        # check dimensions and return
        assert (p.shape == (len(Xstar),)), f"Expected shape for p is ({len(Xstar)}), but the actual shape was {p.shape}. Please check implementation"
        return p
    
    def generate_samples(mean, K, M, jitter=1e-8):
        """ returns M samples from a zero-mean Gaussian process with kernel matrix K
        
        arguments:
        K      -- NxN kernel matrix
        M      -- number of samples (scalar)
        jitter -- scalar
        returns NxM matrix
        """
        
        L = np.linalg.cholesky(K + jitter*np.identity(len(K)))
        zs = np.random.normal(0, 1, size=(len(K), M))
        fs = mean + np.dot(L, zs)
        return fs
    


    def posterior_samples(self, Xstar, num_samples):
        """
            generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar

            Arguments:
                Xstar            -- PxD prediction points
        
            returns:
                f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """
        mu, Sigma = self.predict_f(Xstar)
        f_samples = self.generate_samples(mu.ravel(), Sigma, num_samples)

        assert (f_samples.shape == (len(Xstar), num_samples)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, {num_samples}), but actual shape was {f_samples.shape}. Please check implementation"
        return f_samples


def plot_with_uncertainty(ax, Xp, gp, color='r', color_samples='b', title="", num_samples=0,Xstar=[]):
    
    mu, Sigma = gp.predict_y(Xp)
    mean = mu.ravel()
    std = np.sqrt(np.diag(Sigma))

    # plot distribution
    ax.plot(Xp, mean, color=color, label='Mean')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')
    
    # generate samples
    if num_samples > 0:
        fs = gp.posterior_samples(Xstar, num_samples)
        ax.plot(Xp, fs[:,0], color=color_samples, alpha=.25, label="$f(x)$ samples")
        ax.plot(Xp, fs[:, 1:], color=color_samples, alpha=.25)
    
    ax.set_title(title)

"""
[Xtrain,ytrain,Xtest,ytest,age] = get_data(False,True)
print(Xtrain.shape)


kernel = StationaryIsotropicKernel(squared_exponential)
gpc = GaussianProcessClassification(Xtrain, ytrain, BernoulliLikelihood, kernel, kappa=1, lengthscale=1)

# predict
p_train = gpc.predict_y(Xtrain)
p_test = gpc.predict_y(Xtest)

# make predictions
ytrain_hat = 1.0*(p_train > 0.5)
ytest_hat = 1.0*(p_test > 0.5)

# print results: mean and standard error of the mean
print('Training error:\t%3.2f (%3.2f)' % compute_err(ytrain_hat.ravel(), ytrain.ravel()))
print('Test error:\t%3.2f (%3.2f)' % compute_err(ytest_hat.ravel(), ytest.ravel()))

# evaluate prediction accuracy
print("Accuracy Train:", 1.0*np.sum(ytrain_hat == ytrain) / len(ytrain))
print("Accuracy Test:", 1.0*np.sum(ytest_hat == ytest) / len(ytest))
"""