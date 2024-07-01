import numpy as np
import matplotlib.pyplot as plt
from tools import *

def forward_simulation_1d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mu,mu0, vses=None, gammases=None, returnCA = False):
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

    trans_As = np.zeros((N,5,5))
    noise_Cs = np.zeros((N,5,5))
    
    for n in range(N-1):
        vs, gammas = generate_jumps(c, delta_t)
        # vs = vses[n]
        # gammas = gammases[n]
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
        S_mu = processer.S_mu(sigma_mu)
        S_st = processer.S_s_t(sigma_w)
        m_st = processer.mean_s_t()
        hi_fi_intQ = processer.hi_fi_intQ(sigma_mu, l)
        hi_fi_vi = processer.hi_fi_vi(sigma_mu)
        
        trans_A = transition_matrix(m_st, delta_t, l)
        noise_C = noise_variance_C(S_mu, S_st, hi_fi_intQ, hi_fi_vi, delta_t, l, sigma_mu)
        
        x_dashed[n+1,:] = trans_A@ x_dashed[n,:] + np.random.multivariate_normal([0]*5, noise_C)
        trans_As[n+1,:] = trans_A
        noise_Cs[n+1,:] = noise_C
    
    if returnCA:
        return x_dashed, trans_As, noise_Cs
    else:
        return x_dashed

def forward_simulation_1d(alpha, l, c, N, delta_t, sigma_w, sigma_mu,mu0, return_jumps=False):
    """
    simulate directly by estimating integrals by Reimmann sums, 1d
    """
    x_dashed = np.zeros((N,3))
    x_dashed[0,-1] = mu0
    integrals = np.zeros((N, 2))

    vses = []
    gammases = []
    
    for n in range(N-1):
        vs, gammas, mus = generate_jumps(c, delta_t, sigma_mu)
        mus += x_dashed[n, -1] # mus are from 0, need to add the initial value
        vses.append(vs)
        gammases.append(gammas)
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
        S_st = processer.S_s_t(sigma_w)
        m_st = processer.mean_s_t(mus[:-1])
        integrals[n,:] = processer.int_mu(mus)
        
        x_dashed[n+1,:2] = eAt(l, delta_t)@ x_dashed[n,:2]+ m_st - alpha/(alpha-1) * c**(1-1/alpha)*integrals[n,:] \
            + np.random.multivariate_normal([0]*2, S_st)
        x_dashed[n+1,-1] = mus[-1]
    
    if return_jumps:
        return x_dashed, vses, gammases
    else:
        return x_dashed

def simu(l, c, N, delta_t, sigma_w, sigma_mu, mu0, alpha, k_v, save = False):
    # x_dashed_noint, vses, gammases = forward_simulation_1d(alpha, l, c, N, delta_t, sigma_w, sigma_mu, mu0, return_jumps=True)
    x_dashed = forward_simulation_1d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mu, mu0)
    y_ns = x_dashed[:,:2]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2:4]

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y_ns[:,0])
    plt.ylabel('displacement')
    plt.subplot(2,1,2)
    plt.plot(y_ns[:,1])
    plt.ylabel('velocity')
    plt.xlabel('Time (n)')
    plt.show()
    if save:
        plt.savefig(f'experiments/figure/simulation/xs_a{int(alpha*10)}.png')
    
    y_ns_noisy = y_ns+np.random.normal(0, sigma_w*k_v, (N, 2))
    plt.figure()
    plt.plot(y_ns[:,0])
    plt.scatter(range(N), y_ns_noisy[:,0], color='orange',s=5)
    plt.xlabel('Time (n)')
    plt.legend(['True state', 'Noisy observation'])
    plt.show()
    if save:
        plt.savefig(f'experiments/figure/simulation/ys_a{int(alpha*10)}.png')
        # save
        np.savez('experiments/data/x_ns.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0, c = c, l = l, alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mu, delta_t = delta_t, \
            k_v = k_v, allow_pickle=True)

def simu_pure_noise(mu0, c, delta_t, sigma_mu, alpha, sigma_w, N):
    x_dashed = np.zeros((N,2))
    x_dashed[0,-1] = mu0
    integrals = np.zeros(N)
    
    for n in range(N-1):
        vs, gammas, mus = generate_jumps(c, delta_t, sigma_mu)
        mus += x_dashed[n, -1] # mus are from 0, need to add the initial value
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l=0)
        # import pdb;pdb.set_trace()
        S_st = processer.S_s_t(sigma_w)[0,0]
        m_st = processer.mean_s_t(mus[:-1])[0]
        integrals[n] = processer.int_mu(mus)
        
        x_dashed[n+1] = x_dashed[n]+ m_st - alpha/(alpha-1) * c**(1-1/alpha)*integrals[n] \
            + np.random.normal(0, S_st)
        x_dashed[n+1,-1] = mus[-1]

    return x_dashed
    
if __name__=='__main__':
    l = -0.05
    c = 10
    N = 100
    delta_t = 1
    sigma_w = 0.05
    sigma_mu = 0.15*0
    mu0 = 0.05
    alpha = 1.6
    k_v = 500

    # simu(l, c, N, delta_t, sigma_w, sigma_mu, mu0, alpha, k_v)
    x_finals = np.zeros(400)
    for i in range(400):
        x_ns = simu_pure_noise(mu0, c, delta_t, sigma_mu, alpha, sigma_w, N)
        x_finals[i] = x_ns[-1][0]

    plt.hist(x_finals, bins = 100, range=[-200,200])
    print(np.mean(x_finals))
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(x_ns[:,0])
    # plt.ylabel('displacement')
    # plt.subplot(2,1,2)
    # plt.plot(x_ns[:,1])
    # plt.ylabel('Mu')
    # plt.xlabel('Time (n)')
    plt.show()
    
    # data_read = np.load('experiments/data/x_ns.npz')
    # x_dashed = data_read['x']
    # y_ns = x_dashed[:,:2]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2:4]
