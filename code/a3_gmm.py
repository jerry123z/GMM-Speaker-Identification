from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

#dataDir = '/u/cs401/A3/data/'
dataDir = r'C:\Users\Jerry\Documents\CSC401\A3\data'

class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def pre_compute(m, myTheta):
    mu = myTheta.mu[m]
    sigma = myTheta.Sigma[m]
    M, d = myTheta.Sigma.shape

    term1 = np.sum(np.square(mu) / sigma / 2.0)
    term2 = d / 2.0 * np.log(2.0 * np.pi)
    term3 = 0.5 * np.log(np.prod(sigma))
    return - term1 - term2 - term3


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    precomputed = preComputedForM[m]
    sigma = myTheta.Sigma[m]
    mu = myTheta.mu[m]
    term1 = 0.5 * np.multiply(np.square(x), np.reciprocal(sigma))
    term2 = np.multiply(np.multiply(mu, x), np.reciprocal(sigma))

    first_term = - np.sum(np.subtract(term1, term2), axis=0)
    return first_term + precomputed


def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M, d = myTheta.Sigma.shape
    omega_m = myTheta.omega[m, 0]
    omega_M = myTheta.omega

    preComputedForM = []
    for i in range(M):
        preComputedForM.append(pre_compute(i, myTheta))

    log_b_list = np.zeros(M)
    for i in range(M):
        log_b_list[i] = log_b_m_x(i, x, myTheta, preComputedForM)

    log_omega_list = np.zeros(M)
    for i in range(M):
        log_omega_list[i] = np.log(omega_M[i, 0])

    term1 = np.log(omega_m)
    term2 = log_b_list[m]
    term3 = logsumexp(log_b_list + log_omega_list)
    return term1 + term2 - term3


def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    log_omegas = np.log(myTheta.omega)
    return np.sum(logsumexp(log_Bs + log_omegas, axis=0))


def vector_log_b(m, X, myTheta, preComputedForM=[]):
    precomputed = preComputedForM[m]
    sigma = myTheta.Sigma[m]
    mu = myTheta.mu[m]
    term = np.sum(-0.5 * np.multiply(np.square(X), np.reciprocal(sigma)) + np.multiply(np.multiply(mu, X), np.reciprocal(sigma)), axis=1)
    return term + precomputed

def vector_log_p(log_bs, myTheta):
    log_omegabs = log_bs + np.log(myTheta.omega)
    return log_omegabs - logsumexp(log_omegabs, axis=0)


def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    myTheta = theta( speaker, M, X.shape[1] )
    T, d = X.shape

    prev_L = float('-inf')
    improvement = float('inf')
    myTheta.omega = np.full((M, 1), 1/M)
    myTheta.mu = np.array([X[i] for i in np.random.choice(T, M)])
    myTheta.Sigma = np.full((M, d), 1.0)

    iteration = 0
    while iteration < maxIter and improvement >= epsilon:
        # compute intermediate results
        preComputedForM = []
        for m in range(M):
            preComputedForM.append(pre_compute(m, myTheta))

        # testing non-vercotized version
        #log_bs = np.zeros((M, T))
        #for m in range(M):
        #    for t in range(T):
        #        log_bs[m, t] = log_b_m_x(m, X[t], myTheta, preComputedForM)

        log_bs = np.zeros((M, T))
        for m in range(M):
            log_bs[m] = vector_log_b(m, X, myTheta, preComputedForM)

        # testing non-vectorized version
        #log_ps = np.zeros((M, T))
        #for m in range(M):
        #    for t in range(T):
        #        log_ps[m, t] = log_p_m_x(m, X[t], myTheta)

        log_ps = vector_log_p(log_bs, myTheta)

        # Compute Likelyhood
        L = logLik(log_bs, myTheta)

        # Update Parameters
        p = np.exp(log_ps)
        sum_p = np.sum(np.exp(log_ps), axis=1)  #length 8 vector

        for m in range(M):
            myTheta.omega[m] = np.divide(sum_p[m],T)

        mu_matrix = (np.dot(p, X))
        for m in range(M):
            myTheta.mu[m] = mu_matrix[m] / sum_p[m]
        sigma_matrix = np.dot(p, np.square(X))
        for m in range(M):
            myTheta.Sigma[m] = sigma_matrix[m] / sum_p[m] - np.square(myTheta.mu[m])

        improvement = L - prev_L
        # print(improvement)
        prev_L = L
        iteration = iteration + 1
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    log_likelyhood = np.zeros(len(models))
    T, d = mfcc.shape
    M = models[0].omega.shape[0]

    log_bs = np.zeros((len(models), M, T))
    for i in range(len(models)):
        preComputedForM = []
        for m in range(M):
            preComputedForM.append(pre_compute(m, models[i]))
        for m in range(M):
            log_bs[i, m, :] = vector_log_b(m, mfcc, models[i], preComputedForM)

    for i in range(len(models)):
        log_likelyhood[i] = logLik(log_bs[i], models[i])

    bestModel = np.argmax(log_likelyhood)
    if k > 0:
        top_k = log_likelyhood.argsort()
        print('{}'.format(models[correctID].name))
        for i in range(1, k+1):
            print('{} {}'.format(models[int(top_k[-i])].name, log_likelyhood[int(top_k[-i])]))

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 1
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )

            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print("Accuracy: {}".format(accuracy))
