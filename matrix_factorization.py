
from __future__ import division, print_function
import numpy as np

def sigmoid(M):
    '''
    '''

    return 1 / (1 + np.exp(-np.asarray(M) ))

def linear_cost(params, Y, R, K, lamda):
    '''
    '''
    N, M = Y.shape
    split_idx = N * K
    P = params[:split_idx].reshape(N, K)
    Q = params[split_idx:].reshape(M, K)

    J = 0

    # COST
    diff = R * (np.dot(P,Q.T) - Y)
    #print(diff)
    J += 0.5 * sum(sum(diff ** 2)) / (N * M)
    # REGULARIZATION
    J += lamda / 2 * ( sum(sum(P ** 2)) / (N * K) + sum(sum(Q ** 2)) / (M * K))

    return J

def log_cost(params, Y, R, K, lamda):
    '''
    '''
    N, M = Y.shape
    split_idx = N * K
    P = params[:split_idx].reshape(N, K)
    Q = params[split_idx:].reshape(M, K)

    J = 0

    # COST
    eY = 5 * sigmoid(np.dot(P,Q.T))
    diff = R * (eY - Y)

    J += 0.5 * sum(sum(diff ** 2))
    # REGULARIZATION
    J += lamda / 2 * ( sum(sum(P ** 2)) + sum(sum(Q ** 2)) )

    return J

def log_grad(params, Y, R, K, lamda):
    N, M = Y.shape
    split_idx = N * K
    P = params[:split_idx].reshape(N, K)
    Q = params[split_idx:].reshape(M, K)

    P_grad = np.zeros(P.shape)
    Q_grad = np.zeros(Q.shape)

    # GRAD
    eY = 5 * sigmoid(np.dot(P,Q.T))
    diff = R * (eY - Y)
    P_grad = np.dot(diff, Q)
    Q_grad = np.dot(diff.T, P)

    # REGULARIZATION
    P_grad += lamda * P
    Q_grad += lamda * Q

    grad = np.hstack([P_grad.flatten(),Q_grad.flatten()])

    return grad

def linear_grad(params, Y, R, K, lamda):
    '''
    '''
    N, M = Y.shape
    split_idx = N * K
    P = params[:split_idx].reshape(N, K)
    Q = params[split_idx:].reshape(M, K)

    P_grad = np.zeros(P.shape)
    Q_grad = np.zeros(Q.shape)

    # GRAD
    diff = R * (np.dot(P,Q.T) - Y)
    P_grad = np.dot(diff, Q)
    Q_grad = np.dot(diff.T, P)

    # REGULARIZATION
    P_grad += lamda * P
    Q_grad += lamda * Q

    grad = np.hstack([P_grad.flatten(),Q_grad.flatten()])

    return grad

def grad_descent_factorization(Y, P=None, Q=None, K=10, 
                               lamda=0.01, alpha=0.01, eps0=1e-5,
                               steps=5000, mode='linear', disp=False):
    '''
    Parameters
    ----------
    Y:          N x M Rating Matrix to be factored, P Q.T = Y
    P:          N x K initial guess for P, default is random
    Q:          M x K initial guess for Q, default is random
    K:          the number of latent features
    lamda:      regularization parameter
    alpha:      learning rate
    steps:      max number of steps

    Returns
    -------
    P_min       N x K matrix| Together the product P_min Q_min.T minimizes 
    Q_min       M x K matrix| the cost associated with the factorization of Y.

    Notes:      P.shape == (N,K), Q.shape == (M,K) assert well defined products
                when P,Q, and K are simultaneously specified.
    '''
    np.seterr(all='warn')
    gradCost = linear_grad
    Cost = linear_cost
    N, M = Y.shape
    if P is None:
        P = np.random.rand(N,K)
    if Q is None:
        Q = np.random.rand(M,K)
    assert( Q.shape == (M,K) and P.shape == (N,K) )
    params = np.hstack([P.flatten(),Q.flatten()])
    split_idx = N * K
    P_cur = params[:split_idx].reshape(N, K)
    Q_cur = params[split_idx:].reshape(M, K)

    R = (Y != 0).astype(int)

    step_count = 0
    J_prev = Cost(params, Y, R, K, lamda)
    eY = np.dot(P_cur, Q_cur.T)
    err_prev = max(abs(eY * R - Y).flatten())
    for steps in range(steps):
        grad = gradCost(params, Y, R, K, lamda)
        params += -alpha * grad
        J = Cost(params, Y, R, K, lamda)
        step_count += 1

        if disp and steps % 10 == 0:
            P_cur = params[:split_idx].reshape(N, K)
            Q_cur = params[split_idx:].reshape(M, K)
            eY = np.dot(P_cur, Q_cur.T)
            err = max(abs(eY * R - Y).flatten())
            print("{0}, C={1:.2f}, Err={2:.2f}, CI={3:.3%}, EI={4:.2%}".format(steps,
                                       J, err, J/J_prev, err/err_prev))


        if abs(J_prev - J) < eps0:
            break
        J_prev = J
        
    print("Minimization terminated at step {0}".format(step_count))
    print("Cost Function Value at Termination = {0}".format(J))
    split_idx = N * K
    P_min = params[:split_idx].reshape(N, K)
    Q_min = params[split_idx:].reshape(M, K)

    return P_min, Q_min

if __name__ == "__main__":

    Y = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])
    N, M = Y.shape
    K = 10
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    print(sigmoid(0))
    print(sigmoid([5, 0]))

    print("GRADIENT DESCENT")
    nP, nQ = grad_descent_factorization(Y,K=10)
    eY = np.dot(nP, nQ.T)
    print("Original Matrix\n",Y)
    print("Reconstructed Estimated Matrix\n",eY)
    print("Learned Ratings\n", eY * (Y == 0).astype(int))
    

    print(max(abs(eY * (Y != 0).astype(int) - Y).flatten()) )
    


        
