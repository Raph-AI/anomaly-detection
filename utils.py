import numpy as np
import math
import multiprocessing
import collections
import iisignature
from iisignature import sigcombine


def p_var_backbone(path_size, p, path_dist):
    """
    Input:
    * path_size >= 0 integer
    * p >= 1 real
    * path_dist: metric on the set {0,...,path_dist-1}.
      Namely, path_dist(a,b) needs to be defined and nonnegative
      for all integer 0 <= a,b < path_dist, be symmetric and
      satisfy the triangle inequality:
      * path_dist(a,b) = path_dist(b,a)
      * path_dist(a,b) + path_dist(b,c) >= path_dist(a,c)
      Indiscernibility is not necessary, so path_dist may not
      be a metric in the strict sense.
    Output: a class with two fields:
    * .p_var = max sum_k path_dist(a_{k-1}, a_k)^p
               over all strictly increasing subsequences a_k of 0,...,path_size-1
    * .points = the maximising sequence a_k
    Notes:
    * if path_size == 0, the result is .p_var = -math.inf, .points = []
    * if path_size == 1, the result is .p_var = 0,         .points = [0]

    Credits: <https://github.com/khumarahn/p-var>
    """

    ret = collections.namedtuple('p_var', ['value', 'points'])

    if path_size == 0:
        return ret(value = -math.inf, points = [])
    elif path_size == 1:
        return ret(value = 0, points = [0])

    s = path_size - 1
    N = 1
    while s >> N != 0:
        N += 1

    ind = [0.0] * s
    def ind_n(j, n):
        return (s >> n) + (j >> n)
    def ind_k(j, n):
        return min(((j >> n) << n) + (1 << (n-1)), s);

    max_p_var = 0.0
    run_p_var = [0.0] * path_size

    point_links = [0] * path_size

    for j in range(0, path_size):
        for n in range(1, N + 1):
            if not(j >> n == s >> n and (s >> (n-1)) % 2 == 0):
                ind[ind_n(j, n)] = max(ind[ind_n(j, n)], path_dist(ind_k(j, n), j))
        if j == 0:
            continue

        m = j - 1
        delta = 0.0
        delta_m = j
        n = 0
        while True:
            while n > 0 and m >> n == s >> n and (s >> (n-1)) % 2 == 0:
                n -= 1;

            skip = False
            if n > 0:
                iid = ind[ind_n(m, n)] + path_dist(ind_k(m, n), j)
                if delta >= iid:
                    skip = True
                elif m < delta_m:
                    delta = pow(max_p_var - run_p_var[m], 1. / p)
                    delta_m = m
                    if delta >= iid:
                        skip = True

            if skip:
                k = (m >> n) << n
                if k > 0:
                    m = k - 1
                    while n < N and (k >> n) % 2 == 0:
                        n += 1
                else:
                    break
            else:
                if n > 1:
                    n -= 1
                else:
                    d = path_dist(m, j)
                    if d >= delta:
                        new_p_var = run_p_var[m] + pow(d, p)
                        if new_p_var >= max_p_var:
                            max_p_var = new_p_var
                            point_links[j] = m
                    if m > 0:
                        while n < N and (m >> n) % 2 == 0:
                            n += 1
                        m -= 1
                    else:
                        break
        run_p_var[j] = max_p_var

    points = []
    point_i = s
    while True:
        points.append(point_i)
        if point_i == 0:
            break
        point_i = point_links[point_i]
    points.reverse()
    return ret(value = run_p_var[-1], points = points)


def p_var_norm(path, p):
    """
    Compute the p-variation norm for a multivariate time series.
    
    Reference
    ---------
    <https://en.wikipedia.org/wiki/P-variation#Computation_of_p-variation_for_discrete_time_series>
    """    
    stream, channels = path.shape
    dist = lambda a, b: math.sqrt(sum([pow(path[b][d] - path[a][d], 2) for d in range(channels)]))
    pv = p_var_backbone(stream, p, dist).value
    return(np.power(pv, 1./p))


def datascaling_one(path, p=2):
    pv = p_var_norm(path, p)
    if pv != 0.:
        return(path/pv)
    else:
        return(path)
    

def datascaling(data, p=2):
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cpus)
    pool.map(datascaling_one, data)
    data_scaled = pool.map(datascaling_one, data)
    data_scaled = np.array(data_scaled)
    pool.close()
    pool.join()
    return(data_scaled)


def sigscaling_power(SX, depth, channels):
    """
    Each element of signature tensor of depth m is raised to the power 1/m 
    Caveat: use :func:`utils.datascaling` before computation of signature and 
    before using this function.
    """
    if len(SX.shape) != 2:
        raise ValueError(
            "Input SX should be a batch of signatures, i.e. 2-dim array,"
            f" got {SX.shape} instead. If only one signature to rescale, "
            "reshape your input using `SX.reshape((1,)+SX.shape)`."
        )
    if channels != 1:    
        if SX.shape[1] != channels*(channels**depth-1)//(channels-1):
            raise ValueError("SX shape do not match with (depth, channels) parameters.")
        
    # indices of each signature tensor
    inds = np.cumsum([0]+[channels**k for k in range(1, depth+1)])
    SX_scaled = np.zeros((SX.shape))
    for i in range(len(SX)):
        for d in range(1, depth+1):
            idx1, idx2 = inds[d-1], inds[d]
            # sign = np.abs(SX[i, idx1:idx2])/SX[i, idx1:idx2]
            sign = np.sign(SX[i, idx1:idx2])
            SX_scaled[i, idx1:idx2] = sign*np.power(np.abs(SX[i, idx1:idx2]), 1./d)
    return(SX_scaled)


def add_basepoint(X):
    """
    Add a zero at the beginning of every time series to remove y-axis shifts 
    invariance of the signature transform. <https://arxiv.org/abs/2006.00873>
    """
    zeros_to_add = np.zeros((X.shape[0], 1, X.shape[2]))
    X_augmented = np.concatenate((zeros_to_add, X), axis = 1)
    return X_augmented


def leadlag(X, lag):
    """
    Apply lead-lag transformation [1] to the raw time series. 
    [1] <https://arxiv.org/abs/2006.00873>
    """
    if len(X.shape) != 3:
        raise ValueError("Input data should be 3-dimensional.")    
    batch, stream, channels = X.shape
    stream_aug = stream+lag
    X_augmented = np.zeros((batch, stream_aug, channels*(1+lag)))
    X_augmented[:, :stream, :channels] = X
    X_augmented[:, stream:, :channels] = np.repeat(X[:, -1:, :], lag, axis=1)
    for i in range(1, lag+1):
        X_augmented[:, i:stream+i, channels*i:channels*(i+1)] = X
        if i != lag:
            # duplicate start
            X_augmented[:, :i, channels*i:channels*(i+1)] = np.repeat(X[:, :1, :], i, axis=1)
            # duplicate end
            X_augmented[:, stream+i:, channels*i:channels*(i+1)] = np.repeat(X[:, -1:, :], lag-i, axis=1)
    X_augmented[:, :lag, channels*lag:channels*(lag+1)] = np.repeat(X[:, :1, :], lag, axis=1)
    return X_augmented


def dyadic_sig(X, sig_depth, dyadic_depth):
    """
    Compute the signature on each sliding window of length equal to n*2^(-i+1) up to 
    required depth.
    
    Input
    -----
    X : (batch, stream, channels) nd.array
        Input time series
    sig_depth : int
        The signature is computed up to depth/level `sig_depth`.
    dyadic_depth : int
        Depth of the segmentation.
    
    Output
    ------
    SX : (batch, n_sigs, len_sigs) nd.array
        Signature computed on each sub-interval of the dyadic decomposition of 
        the path. `n_sigs` is the number of subintervals created after the 
        dyadic decomposition 
    """
    batch_X, stream_X, channels_X = X.shape
    n_chunks = 2**dyadic_depth
    bounds_chunks = np.linspace(0, stream_X, n_chunks+1, dtype='int')  
    n_sigs = 2**(dyadic_depth+1)-1
    len_sigs = (channels_X**(sig_depth+1)-channels_X)//(channels_X-1)
    SX = np.zeros((batch_X, n_sigs, len_sigs))
    upper = n_sigs - 2**dyadic_depth
    lower = upper - 2**(dyadic_depth-1)

    for i in range(n_chunks):
        stream_left  = bounds_chunks[i]
        if i != n_chunks-1:
            stream_right = bounds_chunks[i+1]+1
        else:
            stream_right = bounds_chunks[i+1]
        SX[:, upper+i] = iisignature.sig(X[:, stream_left:stream_right, :], sig_depth)

    for i in range(dyadic_depth, 0, -1):
        for j in range(2**(i-1)):
            SX[:, lower+j] = sigcombine(SX[:, upper+2*j], SX[:, upper+2*j+1], channels_X, sig_depth)
        upper = lower
        lower = lower - 2**(i-2)
    return SX