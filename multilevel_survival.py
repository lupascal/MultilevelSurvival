import numpy as np
from typing import Dict, List



def fit(features: np.array,
        groups: Dict[int, List[int]],
        penalty1: float,
        penalty2: float,
        num_gen: int,
        num_im: int,
        epsilon_0: float = 0.01,
        n_iter: int =200) -> np.array:
    """Fit w in a Cox model h(t) = h_0(t) exp(w*x)
    
    Parameters
    ----------
    features: np.array
        must be of size num_gen*num_im + 2, where the two last columns are E, T
    groups: Dict[int, List[int]]
        dictionary where keys are genes and values are SNPs associated to gene
    penalty1: float
        lambda_W
    penalty2: float
        lambda_I
    num_gen: float
        number of SNPs
    num_im: float
        number of MRI features
    epsilon_0: float
    n_iter: int
        number of iterations
    
    Returns
    -------
    w: np.array
        w = [flatten(W), beta_I]
    """
    
    w = np.zeros(features.shape[1]-2)
    w_prev = w
    i = 0
    loss = float('inf')
    norm = 0
    epsilon = epsilon_0
    num_div=1
    index = num_im*num_gen
    
    while True:
        if i >= n_iter:
            break
        if num_div > 20:
            break
        
        w_new = w - epsilon*gradient(features, w)
        
        # matrix W
        if penalty1 >= 0:
            for g in range(len(groups)):
                indices_g = groups[g]
                w_reg = epsilon*penalty1*np.sqrt(len(indices_g))
                W_i_g = w_new[groups[g]]
                norm_W_i_g = np.linalg.norm(W_i_g)
                if (norm_W_i_g > 0):
                    regul = max(1 - w_reg/norm_W_i_g, 0)
                    w_new[indices_g] = regul*W_i_g
                else:
                    w_new[indices_g] = W_i_g
        
        # beta_I
        if penalty2 >= 0:
            threshold2 = epsilon*penalty2
            w_new[index:] = (w_new[index:])/(1+2*epsilon*penalty2)
        if penalty2 < 0:
            w_new[index:] = np.zeros(w_new[index:].shape)
        
        norm_new = penalty2*np.linalg.norm(w_new[index:])**2 + penalty1*np.sum([(np.linalg.norm(w_new[groups[g]])**2)*np.sqrt(len(groups[g])) for g in groups])
        g = 1/epsilon*(w_new - w)
        loss_new = log_likelihood(features, w_new)+norm_new
        
        if loss_new > loss:
            epsilon = epsilon/2
            i += 1
            num_div += 1
            continue
        
        if loss_new == float('nan'):
            epsilon = epsilon/2
            i += 1
            num_div += 1
            continue
    
        w_prev = w
        w = w_new
        epsilon = epsilon_0
        
        res = np.abs(1 - (loss_new / loss))
        
        if (res < 1e-20):
            break

    loss = loss_new
    norm = norm_new
        i += 1
        num_div = 1
    
    return w



def gradient(features: np.array,
             beta: np.array) -> np.array:
    """Compute log-likelihood gradient in a Cox model with respect to beta.
    h(t) = h_0(t) exp(beta*x)
        
    Parameters
    ----------
    features: np.array
        The two last columns are E, T
    beta: np.array
        Vector where gradient is computed
        
    Returns
    -------
    gradient: np.array
    
    """
    
    time = features[:, -1]
    o = np.argsort(-time, kind="mergesort")
    x, _, _ = scaler(features[o, :-2])
    time = np.array([features[o, -1]]).transpose()
    E = np.array([features[o, -2]]).transpose()
    n = features.shape[0]
    betax = np.dot(x, beta)
    
    exp_xw = np.exp(np.dot(x, beta))
    n_samples, n_features = x.shape
    
    gradient = np.zeros((1, n_features), dtype=float)
    
    inv_n_samples = 1. / n_samples
    risk_set = 0
    risk_set_x = 0
    k = 0
    
    # iterate time in descending order
    for i in range(n_samples):
        ti = time[i]
        while k < n_samples and ti == time[k]:
            risk_set += exp_xw[k]
            
            # preserve 2D shape of row vector
            xk = x[k:k + 1]
            risk_set_x += exp_xw[k] * xk
            k += 1
        if E[i]:
            gradient -= (x[i:i + 1] - risk_set_x / risk_set) * inv_n_samples

    return gradient.ravel()



def log_likelihood(features: np.array,
                   beta: np.array) -> float:
    """Compute log-likelihood in a Cox model with respect to beta.
    h(t) = h_0(t) exp(beta*x)
        
    Parameters
    ----------
    features: np.array
        The two last columns are E, T
    beta: np.array
        Vector where the log-likelihood is computed
        
    Returns
    -------
    loss: float
        log-likelihood
    """
    
    time = features[:, -1]
    o = np.argsort(-time, kind="mergesort")
    features_ = features[o, :-2]
    T = np.array([features[o, -1]]).transpose()
    E = np.array([features[o, -2]]).transpose()
    betax = np.dot(features_, beta)
    n_samples, _ = features_.shape
    
    loss = 0
    risk_set = 0
    k = 0
    
    for i in range(n_samples):
        ti = T[i,0]
        while k < n_samples and ti == T[k,0]:
            risk_set += np.exp(betax[k])
            k += 1
        
        if E[i,0]:
            loss -= (betax[i] - np.log(risk_set))/n_samples
    return loss
