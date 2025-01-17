import numpy as np
import matplotlib.pyplot as plt
from tools.tools import *


def forward_simulation_1d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mu,mu0, vses=None, gammases=None, returnCA = False):
    """forward simulation of 1d langevin model driven by alpha stable noise, time changing skew

    Args:
        alpha (float): alpha-stable
        l (float): theta parameter in langevin model, usually negative
        c (float): truncation level
        N (int): total number of points taken
        delta_t (float): time difference between each samples, default=1
        sigma_w (float): variance of W_i
        sigma_mu (float): variance of the brownian motion that drives skew, mean of W_i
        mu0 (float): start of E[W_i]

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
        S_st = processer.S_s_t(sigma_w)
        m_st = processer.mean_s_t()
        trans_A = transition_matrix(m_st, delta_t, l)

        S_mu = processer.S_mu(sigma_mu)
        hi_fi_intQ = processer.hi_fi_intQ(sigma_mu, l)
        hi_fi_vi = processer.hi_fi_vi(sigma_mu)
        
        noise_C = noise_variance_C(S_mu, S_st, hi_fi_intQ, hi_fi_vi, delta_t, l, sigma_mu)
        
        x_dashed[n+1,:] = trans_A@ x_dashed[n,:] + np.random.multivariate_normal([0]*5, noise_C)
        trans_As[n+1,:] = trans_A
        noise_Cs[n+1,:] = noise_C
    
    if returnCA:
        return x_dashed, trans_As, noise_Cs
    else:
        return x_dashed

def forward_simulation_2d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mus, mu0s, returnCA = False):
    """forward simulation of 2d langevin model driven by alpha stable noise, time changing skew

    Args:
        alpha (float): alpha-stable
        l (float): theta parameter in langevin model, usually negative
        c (float): truncation level
        N (int): total number of points taken
        delta_t (float): time difference between each samples
        sigma_w (float): variance of W_i
        sigma_mus (list of float, length 2): variance of the brownian motion that drives skew, mean of W_i
        mu0s (list of float, length 2): start of E[W_i]

    Returns:
        x_ns: of size (N,10), first dim: uncentered Xs (N,2), integrals of mus (N,2), mus (N,1); second dim: (N,5:)
    """
    x_dashed = np.zeros((N,10))
    x_dashed[0, 4] = mu0s[0]
    x_dashed[0, -1] = mu0s[1]

    trans_As = np.zeros((N,10,10))
    noise_Cs = np.zeros((N,10,10))
    
    for n in range(N-1):
        trans_A2D = np.zeros((10,10))
        noise_C2D = np.zeros((10,10))

        vs, gammas = generate_jumps(c, delta_t) # jump of the subordinator, shared by both directions
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
        S_st = processer.S_s_t(sigma_w)
        m_st = processer.mean_s_t()
        trans_A = transition_matrix(m_st, delta_t, l)

        trans_A2D[:5,:5] = trans_A
        trans_A2D[5:,5:] = trans_A

        sigma_mu = sigma_mus[0]
        S_mu = processer.S_mu(sigma_mu)
        hi_fi_intQ = processer.hi_fi_intQ(sigma_mu, l)
        hi_fi_vi = processer.hi_fi_vi(sigma_mu)
        noise_C = noise_variance_C(S_mu, S_st, hi_fi_intQ, hi_fi_vi, delta_t, l, sigma_mu)
        noise_C2D[:5,:5] = noise_C

        sigma_mu = sigma_mus[1]
        S_mu = processer.S_mu(sigma_mu)
        hi_fi_intQ = processer.hi_fi_intQ(sigma_mu, l)
        hi_fi_vi = processer.hi_fi_vi(sigma_mu)
        noise_C = noise_variance_C(S_mu, S_st, hi_fi_intQ, hi_fi_vi, delta_t, l, sigma_mu)
        noise_C2D[5:,5:] = noise_C
        
        x_dashed[n+1,:] = trans_A2D@ x_dashed[n,:] + np.random.multivariate_normal([0]*10, noise_C2D)
        trans_As[n+1,:] = trans_A2D
        noise_Cs[n+1,:] = noise_C2D
    
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
        vs, gammas, mus = generate_jumps(c, delta_t, delta_t, sigma_mu)
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
    x_dashed, vses, gammases = forward_simulation_1d(alpha, l, c, N, delta_t, sigma_w, sigma_mu, mu0, return_jumps=True)
    y_ns = x_dashed[:,:2]
    # x_dashed = forward_simulation_1d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mu, mu0)
    # y_ns = x_dashed[:,:2]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2:4]

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y_ns[:,0])
    plt.ylabel('displacement')
    plt.subplot(2,1,2)
    plt.plot(y_ns[:,1])
    plt.ylabel('velocity')
    if save:
        plt.savefig(f'experiments/figure/simulation/xs_a{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y_ns[:,0])
    plt.ylabel('displacement')
    plt.subplot(2,1,2)
    plt.plot(x_dashed[:,-1])
    plt.ylabel('$\mu')
    plt.xlabel('Time (n)')
    if save:
        plt.savefig(f'experiments/figure/simulation/mus_a{int(alpha*10)}.png')
    
    y_ns_noisy = y_ns+np.random.normal(0, sigma_w*k_v, (N, 2))
    plt.figure()
    plt.plot(y_ns[:,0])
    plt.scatter(range(N), y_ns_noisy[:,0], color='orange',s=5)
    plt.legend(['True state', 'Noisy observation'])
    if save:
        plt.savefig(f'experiments/figure/simulation/ys_a{int(alpha*10)}.png')
        # save
        np.savez('C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/x_ns_test.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0, c = c, l = l, alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mu, delta_t = delta_t, \
            k_v = k_v, allow_pickle=True)

def simu2d(l, c, N, delta_t, sigma_w, sigma_mus, mu0s, alpha, k_v, savepath = None):
    # x_dashed_noint, vses, gammases = forward_simulation_1d(alpha, l, c, N, delta_t, sigma_w, sigma_mu, mu0, return_jumps=True)
    x_dashed = forward_simulation_2d_w_integrals(alpha, l, c, N, delta_t, sigma_w, sigma_mus, mu0s)

    y_ns = np.zeros((N,4))
    y_ns[:,:2] = x_dashed[:,:2]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2:4]
    y_ns[:,2:] = x_dashed[:,5:7]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,7:9]

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y_ns[:,0], y_ns[:,2])
    plt.ylabel('displacement')
    plt.subplot(2,1,2)
    plt.plot(y_ns[:,1], y_ns[:,3])
    plt.ylabel('velocity')
    if savepath:
        plt.savefig(f'experiments/figure/2d/simulation/xs_a{savepath}.png')

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time (n)', fontsize='18')
    ax1.set_ylabel('$X(t_n)$', color=color, fontsize='18')
    ax1.plot(x_dashed[:,0], color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    # ax2.set_ylabel('mean of noise, x', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_dashed[:,4], color=color)
    ax2.tick_params(axis='y')
    ax2.axhline(y=0.0,linestyle='dashed',)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savepath:
        fig.savefig(f'experiments/figure/2d/simulation/x_mu_x_{savepath}.png')

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time (n)', fontsize='18')
    # ax1.set_ylabel('displacement, x', color=color)
    ax1.plot(x_dashed[:,5], color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('$\mu(t_n)$', color=color, fontsize='18')  # we already handled the x-label with ax1
    ax2.plot(x_dashed[:,-1], color=color)
    ax2.tick_params(axis='y')
    ax2.axhline(y=0.0,linestyle='dashed',)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savepath:
        fig.savefig(f'experiments/figure/2d/simulation/x_mu_y_{savepath}.png')

    plt.figure()
    idxes = np.linspace(0,N,endpoint=False, num=30, dtype=np.int32)
    plt.plot(y_ns[:,0], y_ns[:,2])  
    plt.quiver(y_ns[idxes,0],y_ns[idxes,2], x_dashed[idxes,4], x_dashed[idxes,-1])
    plt.legend(['$X(t)$','$\mu(t)$'])
    if savepath:
        plt.savefig(f'experiments/figure/2d/simulation/x_mu_{savepath}.png')
    # plt.show()
    
    y_ns_noisy = y_ns+np.random.normal(0, sigma_w*k_v, (N, 4))
    plt.figure()
    plt.plot(y_ns[:,0], y_ns[:,2])
    plt.scatter(y_ns_noisy[:,0], y_ns_noisy[:,2], color='orange',s=5)
    plt.xlabel('Time (n)')
    plt.legend(['True state', 'Noisy observation'])
    if savepath:
        plt.savefig(f'experiments/figure/2d/simulation/ys_a{savepath}.png')
        # save
        np.savez(f'experiments/data/2d/x_ns{savepath}.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0s, c = c, l = l, alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mus, delta_t = delta_t, \
            k_v = k_v, allow_pickle=True)
    # plt.show()

def simu_pure_noise(mu0, c, delta_t, sigma_mu, alpha, sigma_w, N, drift=0):
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

    x_dashed[:,0] += np.linspace(0,(N-1)*drift,num=N)

    return x_dashed
    
if __name__=='__main__':
    l = -0.05
    c = 10
    N = 500
    delta_t = 1
    sigma_w = 0.01
    sigma_mus = np.array([0.015, 0.015])*1e-4
    mu0s = [0.1, 0.05]
    alpha = 0.9
    k_v = 5e3 # 1.5e4 for alpha=0.9

    simu(l, c, N, delta_t, sigma_w, sigma_mus[0], mu0s[0], alpha, k_v, save=True)

    # simu2d(l, c, N, delta_t, sigma_w, sigma_mus, mu0s, alpha, k_v, savepath = f'{int(alpha*10)}')
    # x_ns = simu_pure_noise(mu0s[0]*0, c, delta_t, 1e-4, alpha, sigma_w, N, drift=-0.1)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(x_ns[:,0])
    # plt.ylabel('displacement')
    # plt.subplot(2,1,2)
    # plt.plot(x_ns[:,1])
    # plt.ylabel('Mu')
    # plt.xlabel('Time (n)')
    # plt.show()

    
    
    # data_read = np.load('experiments/data/x_ns.npz')
    # x_dashed = data_read['x']
    # y_ns = x_dashed[:,:2]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2:4]
