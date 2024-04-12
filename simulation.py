import numpy as np
import matplotlib.pyplot as plt
from tools import *

def forward_simulation_1d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mu,mu0):
    """forward simulation of 1d langevin model driven by alpha stable noise, time changing skew

    Args:
        alpha (float): alpha-stable
        l (float): theta parameter in langevin model, usually negative
        c (_type_): truncation level
        N (_type_): total number of points taken
        delta_t (_type_): time difference between each samples
        sigma_w (_type_): variance of W_i
        sigma_mu (_type_): variance of the brownian motion that drives skew, mean of W_i
        mu0 (_type_): start of E[W_i]

    Returns:
        x_ns: of size (N,5), uncentered Xs (N,2), integrals of mus (N,2), mus (N,1)
    """
    x_dashed = np.zeros((N,5))
    x_dashed[0,-1] = mu0
    
    for n in range(N-1):
        vs, gammas = generate_jumps(c, delta_t)
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
        S_mu = processer.S_mu(sigma_mu)
        S_st = processer.S_s_t(sigma_w)
        m_st = processer.mean_s_t()
        
        trans_A = transition_matrix(m_st, delta_t, l)
        noise_C = noise_variance_C(S_mu, S_st, delta_t, l, sigma_mu)
        
        x_dashed[n+1,:] = trans_A@ x_dashed[n,:] + np.random.multivariate_normal([0]*5, noise_C)
    
    return x_dashed

def forward_simulation_1d(alpha, l, c, N, delta_t, sigma_w, sigma_mu,mu0):
    """
    simulate directly by estimating integrals by Reimmann sums, 1d
    """
    x_dashed = np.zeros((N,3))
    x_dashed[0,-1] = mu0
    integrals = np.zeros((N, 2))
    
    for n in range(N-1):
        vs, gammas, mus = generate_jumps(c, delta_t, sigma_mu)
        mus += x_dashed[n, -1] # mus are from 0, need to add the initial value
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
        S_st = processer.S_s_t(sigma_w)
        m_st = processer.mean_s_t(mus[:-1])
        integrals[n,:] = processer.int_mu(mus)
        
        x_dashed[n+1,:2] = eAt(l, delta_t)@ x_dashed[n,:2]+ m_st - alpha/(alpha-1) * c**(1-1/alpha)*integrals[n,:]*(alpha>1) \
            + np.random.multivariate_normal([0]*2, S_st)
        x_dashed[n+1,-1] = mus[-1]
    
    return x_dashed

def constant_mu_simu():
    l = -0.05
    c = 10
    N = 500
    delta_t = 1
    sigma_w = 0.05
    sigma_mu = 0.015*1e-3
    mu0 = 0.05
    alpha = 1.6
    k_v = 700*4 # 10
    
    x_dashed = forward_simulation_1d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mu, mu0)
    y_ns = x_dashed[:,:2]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2:4]*(alpha>1)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y_ns[:,0])
    plt.plot(x_dashed[:,0])
    plt.plot(- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2])
    plt.ylabel('displacement')
    plt.legend(['centered','un-centered','centering term'])
    plt.subplot(2,1,2)
    plt.plot(y_ns[:,1])
    plt.plot(x_dashed[:,1])
    plt.plot(- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,3])
    plt.ylabel('velocity')
    plt.savefig('experiments/figure/const_mu005_x.png')
    
    y_ns += np.random.normal(0, sigma_w*k_v, (N, 2))
    # save
    np.savez('experiments/data/x_ns.npz',x = x_dashed, y=y_ns, mu0 = 0.05, c = 10, l = -0.05, alpha = 1.6, sigma_w = 0.05, sigma_mu = 0.015e-3, allow_pickle=True)

    
if __name__=='__main__':
    l = -0.05
    c = 10
    N = 500
    delta_t = 1
    sigma_w = 0.05
    sigma_mu = 0.015*1e-3
    mu0 = 0.05
    alpha = 1.6
    k_v = 700*4 # 10
    
    constant_mu_simu()
    
    data_read = np.load('experiments/data/x_ns.npz')
    x_dashed = data_read['x']

