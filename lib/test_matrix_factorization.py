from __future__ import division, print_function
import numpy as np
import scipy.optimize as opt

def test_cost():

    num_users = 4
    num_movies = 5
    num_features = 3

    X = np.array([[1.048686,  -0.400232,   1.194119],
                  [0.780851,  -0.385626,   0.521198],
                  [0.641509,  -0.547854,  -0.083796],
                  [0.453618,  -0.800218,   0.680481],
                  [0.937538,   0.106090,   0.361953]])

    Theta = np.array([[0.28544,  -1.68427,   0.26294],
                      [0.50501,  -0.45465,   0.31746],
                      [-0.43192,  -0.47880,   0.84671],
                      [0.72860,  -0.27189,   0.32684]])

    Y = np.array([[5, 4, 0, 0],
                  [3, 0, 0, 0],
                  [4, 0, 0, 0],
                  [3, 0, 0, 0],
                  [3, 0, 0, 0]])
    R = np.array( (Y != 0).astype(int) )

    print("X:\n",X)
    print("Theta:\n",Theta)
    print("Y:\n",Y)
    print("R:\n",R)

    params = np.hstack([X.flatten(),Theta.flatten()])
    print("unrolled X,Theta:\n",params)

    print("Cost Function Output FIRST CALL:")
    J, grad = cost_function(params, Y, R, num_users, num_movies, num_features, 2.0)
    print("J: ", J, "\ngrad:\n", grad)

    print("Cost Function Output SECOND CALL:")
    J = cost(params, Y, R, num_users, num_movies, num_features, 2.0)
    grad = grad_cost(params, Y, R, num_users, num_movies, num_features, 2.0)
    print("J: ", J, "\ngrad:\n", grad)

def cost(params, Y, R, num_features, lamda):
    '''
    '''
    N, M = Y.shape
    split_idx = N * K
    P = params[:split_idx].reshape(N, K)
    Q = params[split_idx:].reshape(M, K)

    J = 0

    # COST
    diff = R * (np.dot(P,Q.T) - Y)
    J += 0.5 * sum(sum(diff ** 2))
    # REGULARIZATION
    J += lamda / 2 * ( sum(sum(P ** 2)) + sum(sum(Q ** 2)) )

    return J

def grad_cost(params, Y, R, num_features, lamda):
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

def scipy_newtoncg(Y, P, Q, K, lamda=0.02):

    N, M = Y.shape
    R = (Y != 0).astype(int)

    params0 = np.hstack([P.flatten(),Q.flatten()])
    print(params0)

    options = {'maxiter' : None,
               'disp': True}

    args = (Y, R, K, lamda)

    result = opt.minimize(cost, params0, jac=grad_cost, \
                         args=args, method='Newton-CG', options=options)
    params_min = result.x
    print(params_min)
    split_idx = N * K
    P_min = params_min[:split_idx].reshape(N, K)
    Q_min = params_min[split_idx:].reshape(M, K)

    return P_min, Q_min

def grad_descent(Y, P, Q, K, lamda=0.02, alpha=0.001, steps=5000):

    N, M = Y.shape
    R = (Y != 0).astype(int)

    params = np.hstack([P.flatten(),Q.flatten()])
    print(params)
    
    for steps in range(steps):
        grad = grad_cost(params, Y, R, K, lamda)
        params += -alpha * grad
        J = cost(params, Y, R, K, lamda)
        if J < 0.001:
            break

    split_idx = N * K
    P_min = params[:split_idx].reshape(N, K)
    Q_min = params[split_idx:].reshape(M, K)

    return P_min, Q_min

def matrix_factorization_vec(Y, P, Q, K, steps=5000, alpha=0.001, lamda=0.02):

    R = (Y != 0).astype(int)

    for step in range(steps):
        
        diff = R * (np.dot(P,Q.T) -Y)
        P_old = np.copy(P)
        P = P - alpha * (np.dot(diff, Q) + lamda * P)
        Q = Q - alpha * (np.dot(diff.T, P_old) + lamda * Q)

        E = R * (np.dot(P,Q.T) - Y)
        e = sum(sum(E ** 2)) + lamda * (sum(sum(P ** 2)) + sum(sum(Q ** 2)))
        e *= 0.5
        if e < 0.001:
            break

    return P, Q



if __name__ == "__main__":

    Y = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])

    N, M = Y.shape
    K = 3
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    print("INITIAL GUESS FOR P and Q")
    print(P)
    print(Q)

    print("GRADIENT DESCENT")
    nP, nQ = grad_descent(Y, P, Q, K)
    print(np.dot(nP, nQ.T))

    print("GRADIENT DESCENT no func")
    nP, nQ = matrix_factorization_vec(Y, P, Q, K)
    print(np.dot(nP, nQ.T))

    print("SCIPY NEWTON-CG")
    nP, nQ = scipy_newtoncg(Y, P, Q, K)
    print(np.dot(nP, nQ.T))
    



