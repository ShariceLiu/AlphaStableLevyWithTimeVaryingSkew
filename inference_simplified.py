import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats
from tools.tools_sim import *
import scipy
# from simulation import forward_simulation_1d_w_integrals
from scipy.stats import invgamma
from scipy.special import logsumexp
from tqdm import tqdm
from real_data import extract_track, extract_mat_data
import pandas as pd

from numpy.random import default_rng

from matplotlib import cm

def resample(log_weight_p, mu_p, var_p, E_ns):
    """resample particles, input/output arrays"""
    P = len(log_weight_p)
    indices = np.random.choice(list(range(P)), size = P, p = np.exp(log_weight_p))
    log_weight_p = np.log(np.ones(P)*(1/P))
    return log_weight_p, mu_p[indices,:], var_p[indices,:,:], E_ns[indices]


def resample_all(log_weight_p, mu_p, var_p, E_ns, n):
    """resample particles, input/output arrays"""
    n_log_w = log_weight_p[n]
    P = len(n_log_w)
    indices = np.random.choice(list(range(P)), size = P, p = np.exp(n_log_w))
    log_weight_p = log_weight_p[:, indices]
    log_weight_p[n] = np.log(np.ones(P)*(1/P))

    return log_weight_p, mu_p[:,indices,:], var_p[:,indices,:,:], E_ns[:,indices]

def kalman_filter(A, C, obs_matrix, noise_sig, mu, var, y_n):
    """
    consider state transition and observation like: 
    x_t = A x_s + e_1, e_1 ~ N(0, C)
    y_t = obs_matrix x_t + e_2, e_2 ~ N(0, noise_sig^2)
    """
    mu_n_prev_n = A@mu
    var_n_prev_n = A@var@A.T + C
    y_hat_n_prev_n = obs_matrix@mu_n_prev_n
    sigma_n_prev_n = obs_matrix@var_n_prev_n @ obs_matrix.T + noise_sig**2
    K = var_n_prev_n @ obs_matrix.T /sigma_n_prev_n
    mu_n_n = mu_n_prev_n + K* (y_n - y_hat_n_prev_n)
    var_n_n = (np.identity(len(mu))-np.outer(K,obs_matrix))@var_n_prev_n

    # if sigma_n_prev_n<=0 or var_n_n[1,1]<0 or var_n_n[-1,-1]<0:
    #     import pdb;pdb.set_trace()
    
    return mu_n_n, var_n_n, sigma_n_prev_n, y_hat_n_prev_n

def kalman_filter2d(A, C, obs_matrix, noise_sig, mu, var, y_n,dim=2):
    """
    any dim>=2
    consider state transition and observation like: 
    x_t = A x_s + e_1, e_1 ~ N(0, C)
    y_t = obs_matrix x_t + e_2, e_2 ~ N(0, noise_sig^2*I)
    """
    mu_n_prev_n = A@mu
    var_n_prev_n = A@var@A.T + C
    y_hat_n_prev_n = obs_matrix@mu_n_prev_n
    sigma_n_prev_n = obs_matrix@var_n_prev_n @ obs_matrix.T + noise_sig**2 * np.identity(dim)
    if (np.linalg.inv(sigma_n_prev_n) == np.inf).any():
        import pdb;pdb.set_trace()
    K = var_n_prev_n @ obs_matrix.T @ np.linalg.inv(sigma_n_prev_n)
    mu_n_n = mu_n_prev_n + K@ (y_n - y_hat_n_prev_n)
    var_n_n = (np.identity(len(mu))-K@obs_matrix)@var_n_prev_n
    # import pdb;pdb.set_trace()
    
    return mu_n_n, var_n_n, sigma_n_prev_n, y_hat_n_prev_n

def simu_1d(save = True):
    l = -0.1
    c = 10
    N = 500
    T = 1
    delta_t = 1
    sigma_w = 0.01
    sigma_mu = 0.015
    mu0 = 0.1
    alpha = 1.4
    k_v = 50

    x_dashed = np.zeros((N,3))
    x_dashed[0,-1] = mu0

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    # Cs=[]
    # As=[]

    for n in range(N-1):
        delta_t_n = delta_ts[n]

        vs, gammas, mus = generate_jumps(c, T, delta_t_n, sigma_mu)
        mus += x_dashed[n, -1] # mus are from 0, need to add the initial value
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t_n, c, T, l)
             
        # transition matrices
        C = noise_variance_C(processer, sigma_w, sigma_mu)
        A = transition_matrix(processer)

        # Cs.append(C)
        # As.append(A)
        
        x_dashed[n+1,:] = A@ x_dashed[n,:]+ np.random.multivariate_normal([0]*3, C)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x_dashed[:,0])
    plt.ylabel('displacement')
    plt.subplot(3,1,2)
    plt.plot(x_dashed[:,1])
    plt.ylabel('velocity')
    plt.subplot(3,1,3)
    plt.plot(x_dashed[:,2])
    plt.ylabel('Mu')

    y_ns_noisy = x_dashed[:,:2] + np.random.normal(0, sigma_w*k_v, (N, 2))

    if save:
        plt.savefig(f'experiments/figure/simulation/simplified/xs_a{int(alpha*10)}.png')
        np.savez('C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/x_ns_test.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0, c = c, l = l, alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mu, delta_t = delta_t, \
            k_v = k_v, allow_pickle=True)

    return x_dashed

def simu_2d(save = True, alpha = 0.9, k_v = 5e4, sigma_w = 0.1):
    l = -0.05
    c = 10
    N = 500
    T = 1
    delta_t = 1
    # sigma_w = 0.1
    sigma_mus = np.array([0.15,0.15])*1e-5
    mu0s = [0.1, 0.05]
    # k_v = 5e4 # 1.6-> 1e3

    x_dashed = np.zeros((N,6))
    x_dashed[0,2] = mu0s[0]
    x_dashed[0,-1] = mu0s[1]

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    # Cs=[]
    # As=[]

    for n in range(N-1):
        delta_t_n = delta_ts[n]

        vs, gammas, mus1 = generate_jumps(c, T, delta_t_n, sigma_mus[0])
        # mus2 = generate_mus(vs, sigma_mus[1], delta_t_n)
        # mus1 += x_dashed[n, 2] # mus are from 0, need to add the initial value
        # mus2 += x_dashed[n, -1]
        
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t_n, c, T, l)
             
        # transition matrices
        C1 = noise_variance_C(processer, sigma_w, sigma_mus[0])
        C2 = noise_variance_C(processer, sigma_w, sigma_mus[0])
        A1 = transition_matrix(processer)

        C, A = np.zeros((6,6)), np.zeros((6,6))

        C[:3,:3] = C1
        C[3:,3:] = C2
        A[:3,:3] = A1
        A[3:, 3:] = A1

        # Cs.append(C)
        # As.append(A)
        
        x_dashed[n+1,:] = A@ x_dashed[n,:]+ np.random.multivariate_normal([0]*6, C)
        # x_dashed[n+1,2] = mus1[-1]
        # x_dashed[n+1,-1] = mus2[-1]

    plt.figure()
    plt.ylabel('displacement')
    plt.plot(x_dashed[:,0], x_dashed[:,3], label='X(t)')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    plt.quiver(x_dashed[quiver_idx,0], x_dashed[quiver_idx,3], x_dashed[quiver_idx,2], x_dashed[quiver_idx,-1], label=r'$\mu(t)$')
    plt.plot(x_dashed[0,0], x_dashed[0,1], 'go', label='Start')
    plt.legend()
    plt.savefig(f'experiments/figure/simulation/simplified/2d_x_mu_{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x_dashed[:,0], x_dashed[:,3])
    plt.ylabel('displacement')
    plt.subplot(3,1,2)
    plt.plot(x_dashed[:,1],x_dashed[:,4])
    plt.ylabel('velocity')
    plt.subplot(3,1,3)
    plt.plot(x_dashed[:,2], x_dashed[:,-1])
    plt.ylabel('Mu')
    if save:
        plt.savefig(f'experiments/figure/simulation/simplified/xs_a{int(alpha*10)}.png')

    y_ns_noisy = x_dashed[:,(0,3)] + np.random.normal(0, sigma_w*k_v, (N, 2))

    plt.figure()
    plt.plot(x_dashed[:,0], x_dashed[:,3])
    plt.scatter(y_ns_noisy[:,0], y_ns_noisy[:,1], color='orange',s=5)

    if save:
        plt.savefig(f'experiments/figure/simulation/simplified/ys_a{int(alpha*10)}.png')
        np.savez(f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/2d/x_ns_{int(alpha*10)}.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0s, c = c, l = l, alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mus, delta_t = delta_t, \
            k_v = k_v, allow_pickle=True)

    return x_dashed

def simu_2d_w_drift(save = True, alpha = 0.9, k_v = 5e4, sigma_w = 0.1):
    l = -0.05
    c = 10
    N = 500
    T = 1
    delta_t = 1
    sigma_mus = np.array([0.15,0.15])*1e-1
    sigma_betas = np.array([0.15,0.15])*1e-5
    mu0s = [0.1, 0.05]
    beta0s = [0.1 , 0.05]
    D=4

    x_dashed = np.zeros((N,D*2))
    x_dashed[0,2] = mu0s[0]
    x_dashed[0,2+D] = mu0s[1]
    x_dashed[0,3] = beta0s[0]
    x_dashed[0,3+D] = beta0s[1]

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    for n in range(N-1):
        delta_t_n = delta_ts[n]

        vs, gammas, mus1 = generate_jumps(c, T, delta_t_n, sigma_mus[0])
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t_n, c, T, l)
             
        # transition matrices
        C1 = noise_variance_C_w_drift(processer, sigma_w, sigma_mus[0],sigma_betas[0])
        C2 = noise_variance_C_w_drift(processer, sigma_w, sigma_mus[1],sigma_betas[1])
        A1 = transition_matrix_w_drift(processer)

        C, A = np.zeros((D*2,D*2)), np.zeros((D*2,D*2))

        C[:D,:D] = C1
        C[D:,D:] = C2
        A[:D,:D] = A1
        A[D:, D:] = A1

        # Cs.append(C)
        # As.append(A)
        
        x_dashed[n+1,:] = A@ x_dashed[n,:]+ np.random.multivariate_normal([0]*(D*2), C)
        # x_dashed[n+1,2] = mus1[-1]
        # x_dashed[n+1,-1] = mus2[-1]

    plt.figure()
    plt.ylabel('displacement')
    plt.plot(x_dashed[:,0], x_dashed[:,D], label='X(t)')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    plt.quiver(x_dashed[quiver_idx,0], x_dashed[quiver_idx,D], x_dashed[quiver_idx,2], x_dashed[quiver_idx,D+2], label=r'$\mu(t)$', color = 'red',alpha=0.3)
    plt.quiver(x_dashed[quiver_idx,0], x_dashed[quiver_idx,D], x_dashed[quiver_idx,3], x_dashed[quiver_idx,3+D], label=r'$\beta(t)$', color = 'blue',alpha=0.3)
    plt.plot(x_dashed[0,0], x_dashed[0,1], 'go', label='Start')
    plt.legend()
    plt.savefig(f'experiments/figure/wdrift/simu_c_drift/2d_x_mu_{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x_dashed[:,0], x_dashed[:,D])
    plt.ylabel('displacement')
    plt.subplot(2,2,2)
    plt.plot(x_dashed[:,1],x_dashed[:,D+1])
    plt.ylabel('velocity')
    plt.subplot(2,2,3)
    plt.plot(x_dashed[:,2], x_dashed[:,D+2])
    plt.ylabel('Mu')
    plt.subplot(2,2,4)
    plt.plot(x_dashed[:,3], x_dashed[:,D+3])
    plt.ylabel('Drift')
    if save:
        plt.savefig(f'experiments/figure/wdrift/simu_c_drift/xs_a{int(alpha*10)}.png')

    y_ns_noisy = x_dashed[:,(0,D)] + np.random.normal(0, sigma_w*k_v, (N, 2))

    plt.figure()
    plt.plot(x_dashed[:,0], x_dashed[:,D])
    plt.scatter(y_ns_noisy[:,0], y_ns_noisy[:,1], color='orange',s=5)

    if save:
        plt.savefig(f'experiments/figure/wdrift/simu_c_drift/ys_a{int(alpha*10)}.png')
        np.savez(f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/w_drift/x_ns_{int(alpha*10)}.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0s, c = c, l = l, \
                 alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mus, sigma_betas = sigma_betas,delta_t = delta_t, \
                k_v = k_v, allow_pickle=True)

    return x_dashed

def simu_1d_w_drift(save = True, alpha = 0.9, k_v = 5e4, sigma_w = 0.1):
    l = -0.05
    c = 10
    N = 500
    T = 1
    delta_t = 1
    sigma_mus = np.array([0.15,-1])*1e-1
    sigma_betas = np.array([0.15,-1])*1e-5
    mu0s = [0.1, -1]
    beta0s = [0.1 , -1]
    D=4

    x_dashed = np.zeros((N,D))
    x_dashed[0,2] = mu0s[0]
    x_dashed[0,3] = beta0s[0]

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    for n in range(N-1):
        delta_t_n = delta_ts[n]

        vs, gammas, mus1 = generate_jumps(c, T, delta_t_n, sigma_mus[0])
        processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t_n, c, T, l)
             
        # transition matrices
        C = noise_variance_C_w_drift(processer, sigma_w, sigma_mus[0],sigma_betas[0])
        A = transition_matrix_w_drift(processer)

        # Cs.append(C)
        # As.append(A)
        
        x_dashed[n+1,:] = A@ x_dashed[n,:]+ np.random.multivariate_normal([0]*(D), C)
        # x_dashed[n+1,2] = mus1[-1]
        # x_dashed[n+1,-1] = mus2[-1]

    plt.figure()
    plt.ylabel('displacement')
    plt.plot(x_dashed[:,0], label='X(t)')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    plt.quiver(quiver_idx, x_dashed[quiver_idx,0], 0.0, x_dashed[quiver_idx,2], label=r'$\mu(t)$', color = 'black',alpha=0.7)
    plt.quiver(quiver_idx, x_dashed[quiver_idx,0], 0.0, x_dashed[quiver_idx,3], label=r'$\beta(t)$', color = 'red',alpha=0.3)
    plt.legend()
    plt.savefig(f'experiments/figure/wdrift/simu_c_drift/1d_x_mu_{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x_dashed[:,0])
    plt.ylabel('displacement')
    plt.subplot(2,2,2)
    plt.plot(x_dashed[:,1])
    plt.ylabel('velocity')
    plt.subplot(2,2,3)
    plt.plot(x_dashed[:,2])
    plt.ylabel('Mu')
    plt.subplot(2,2,4)
    plt.plot(x_dashed[:,3])
    plt.ylabel('Drift')
    if save:
        plt.savefig(f'experiments/figure/wdrift/simu_c_drift/1d_xs_a{int(alpha*10)}.png')

    y_ns_noisy = x_dashed[:,(0,D)] + np.random.normal(0, sigma_w*k_v, (N, 2))

    plt.figure()
    plt.plot(x_dashed[:,0])
    plt.scatter(y_ns_noisy[:,0], color='orange',s=5)

    if save:
        plt.savefig(f'experiments/figure/wdrift/simu_c_drift/1d_ys_a{int(alpha*10)}.png')
        np.savez(f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/w_drift/1d_x_ns_{int(alpha*10)}.npz',x = x_dashed, y=y_ns_noisy, mu0 = mu0s, c = c, l = l, \
                 alpha = alpha, sigma_w = sigma_w, sigma_mu = sigma_mus, sigma_betas = sigma_betas,delta_t = delta_t, \
                k_v = k_v, allow_pickle=True)

    return x_dashed

def trimmed_mean(x, trim=0.1):
    x = np.sort(x)
    k = max(1, int(trim*len(x)))
    return x[k:-k].mean()

def particle_filter_1d(y_ns, x_ns, P, c, T, sigma_mu, sigma_w, noise_sig, alpha,l, delta_t,  K=16,Cs = None, As = None, sigma_mu0 = 0.0, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001,step=5):
    """
    P: number of particles
    T: 
    """
    N = len(y_ns)

    # step = 3
    
    n_mu0 = np.array([0.0,0.0,sigma_mu0])
    n_mus = np.array([np.array([n_mu0]*P)]*N)
    # n_mus = np.zeros((N, P, 3))
    n_vars = np.zeros((N, P, 3, 3))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    # var_0 = np.identity(3)*500*sigma_w**2
    var_0 = np.identity(3)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 3, 3)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y

    y_nstep_preds = np.zeros(N)
    y_1step_preds = np.zeros(N)
    
    # some useful constants
    observation_matrix = np.array([1,0,0])

    log_marg = 0.0

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    mse = 0.0
    mse1 = 0.0

    hr = 0.0
    hr1 = 0.0

    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            # import pdb;pdb.set_trace()
            # n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
            n_log_ws, n_mus, n_vars, E_ns = resample_all(n_log_ws, n_mus, n_vars, E_ns, n)
        
        delta_t_n = delta_ts[n]
        log_likes = np.zeros(P)

        y_hat1 = np.zeros(P)
        y_hat = np.zeros(P)
        # update
        temps = np.zeros((K,P))
        for p in range(P):
            vs, gammas = generate_jumps(c, T, delta_t_n)
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t_n, c, T, l)
            
            
            # transition matrices
            C = noise_variance_C(processer, sigma_w, sigma_mu)
            A = transition_matrix(processer)

            # C = Cs[n]
            # A = As[n]
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n)
            
            
            # for later prediction
            # step=1
            # K=16
            # vses, gammases = []
            if m<N-1:
                temp_mean=0.0
                for k in range(K):
                    temp_k = n_mus[m,p,:].copy()
                    dtm = sum(delta_ts[n+1:n+2])
                    vs, gammas = generate_jumps(c, T, dtm)
                    # vses.append[vs]
                    # gammases.append[gammas]
                    processer = alphaStableJumpsProcesser(gammas, vs, alpha, dtm, c, T, l)
                    A = transition_matrix(processer)

                    temp_k = (A@temp_k)
                    temps[k,p] = observation_matrix @ temp_k
                    temp_mean += observation_matrix @ temp_k
                y_hat1[p] = np.median(temps[:,p])
                # y_hat1[p] = trimmed_mean(temps[:,p],0.1)
                
                
            # if m < N-1:
            #     vs, gammas = generate_jumps(c, T, delta_ts[n+1])
            #     processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_ts[n+1], c, T, l)
            #     A = transition_matrix(processer)
                
            #     # A1 = transition_matrix_nojump(l, delta_ts[n+1])
            #     # import pdb;pdb.set_trace()
            #     temp = (A@n_mus[m,p,:])
            #     y_hat1[p] = observation_matrix @ temp

            if m < N- step:
                # average over a few draws
                temps_s = np.zeros(K)
                for k in range(K):
                    temp_k = n_mus[m,p,:].copy()
                    dtm = sum(delta_ts[n+1:n+1+step])
                    vs, gammas = generate_jumps(c, T, dtm)
                    processer = alphaStableJumpsProcesser(gammas, vs, alpha, dtm, c, T, l)
                    A = transition_matrix(processer)

                    temp_k = (A@temp_k)
                    temps_s[k] = observation_matrix @ temp_k
                # y_hat[p] = trimmed_mean(temps_s,0.1)
                y_hat[p] = np.median(temps_s)

                    # for i in range(step-1):
                    #     vs, gammas = generate_jumps(c, T, delta_ts[n+2+i])
                    #     processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_ts[n+2+i], c, T, l)
                    #     A = transition_matrix(processer)
                    #     # A = transition_matrix_nojump(l, delta_ts[n+2+i])

                    #     temp = (A@temp)

                    # y_hat[p] = observation_matrix@ temp
                    

            # mse
            # temp = (A@n_mus[m,p,:])
            # y_hat1[p] = observation_matrix @ temp
            # for i in range(step):
            #     temp = (A@temp)

            # y_hat[p] = observation_matrix@ temp

            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            # E_ns[m,p] = -(y_n-y_hat_n_prev_n)**2/(norm_sigma_n_prev_n)/2
            E_ns[m,p] = E_ns[n,p] -(y_n-y_hat_n_prev_n)**2/(norm_sigma_n_prev_n)/2
            beta_w_post_p = beta_w - E_ns[m,p]

            if norm_sigma_n_prev_n <= 0.0:
                import pdb;pdb.set_trace()
            log_like = -0.5*np.log(norm_sigma_n_prev_n)-(alpha_w+m/2)*np.log(beta_w_post_p)\
                    +((m-1)/2+alpha_w)*np.log(beta_w - E_ns[n,p]) +\
                    scipy.special.loggamma(m/2+alpha_w)-scipy.special.loggamma(n/2+alpha_w) # -1/2*np.log(2*np.pi)
            
            n_log_ws[m,p] = n_log_ws[n, p]+ log_like

            log_likes[p] = log_like

        # normalise weights
        if P==1:
            log_marg += -0.5*np.log(norm_sigma_n_prev_n)
            # log_marg += log_like
        else:
            # log_marg += np.log(np.exp(n_log_ws[m,:]).sum())
            log_marg += logsumexp(n_log_ws[m,:])-logsumexp(n_log_ws[n,:])
        n_log_ws[m,:] = n_log_ws[m,:]- logsumexp(n_log_ws[m,:])

        
        if m < N-step:
            y_nstep_preds[m+step] = np.dot(y_hat,np.exp(n_log_ws[m, :]))
            mse += (y_nstep_preds[m+step] - x_ns[m+step])**2
            hr += (np.sign((y_nstep_preds[m+step] - y_nstep_preds[m])*(x_ns[m+step]-x_ns[m]))==1)

        if m < N-1:
            y_1step_preds[m+1] = np.dot(y_hat1,np.exp(n_log_ws[m, :])) #np.dot(np.mean(temps, axis=0), np.exp(n_log_ws[m,:]))
            mse1 += (y_1step_preds[m+1] - x_ns[m+1])**2
            hr1 += (np.sign((y_nstep_preds[m+1] - y_nstep_preds[m])*(x_ns[m+1]-x_ns[m]))==1)
            # if n>=100 and np.abs(y_1step_preds[m+1]-x_ns[m+1])>0.05:
            #     import pdb;pdb.set_trace()
        # zero step pred
        # mse += (np.dot(n_mus[m,:,0],np.exp(n_log_ws[m, :])) - y_n)**2
    
    # mse = np.dot((y_1step_preds[1:] - y_n))
    if P==1:
        tot_log_marg = log_marg - (alpha_w+m/2)*np.log(beta_w_post_p)
        print(E_ns[-1], beta_w_post_p, log_marg, (alpha_w+m/2)*np.log(beta_w_post_p))
    else:
        tot_log_marg = log_marg

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns, tot_log_marg, mse,mse1, y_1step_preds,y_nstep_preds, hr, hr1


def gaussian_pf_1d(y_ns,x_ns, sigma_w, noise_sig,l, delta_t,  step=5,  Cs = None, As = None, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """
    P: number of particles
    T: 
    """
    N = len(y_ns)
    
    P=1
    n_mu0 = np.array([0.0,0.0])
    n_mus = np.array([np.array([n_mu0]*P)]*N)
    # n_mus = np.zeros((N, P, 3))
    n_vars = np.zeros((N, P, 2, 2))

    # initialize x
    var_0 = np.identity(2)*500*sigma_w**2
    n_vars[0] = np.array([var_0]*P) # size: (P, 2, 2)
    n_vars[1] = np.array([var_0]*P)
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y

    n_log_ws = np.zeros((N, P))

    # one step pred of y
    pred_y = np.zeros(N)
    pred_y3 = np.zeros(N)
    
    # some useful constants
    observation_matrix = np.array([1,0])

    log_marg , log_marg1 = 0.0, 0.0

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    mse, mse3, hr, hr3 = 0.0, 0.0, 0.0, 0.0

    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        delta_t_n = delta_ts[n]
        log_likes = np.zeros(P)

        y_hat = np.zeros(P)
        # update
        for p in range(P):           
            # transition matrices
            C = int_fft(l, delta_t_n)*sigma_w**2
            A = eAt(l, delta_t_n)

            # C = Cs[n]
            # A = As[n]
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n)
            
            # mse
            if m<N-1:
                temp = (eAt(l, delta_ts[n+1])@n_mus[m,p,:])
                y_hat[p] = observation_matrix@ temp
                if m< N-step:
                    for i in range(step-1):
                        temp = (eAt(l, delta_ts[n+2+i])@temp)

                    y_hat3 = observation_matrix@ temp

            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[m,p] = E_ns[n,p]-(y_n-y_hat_n_prev_n)**2/(norm_sigma_n_prev_n)/2
            beta_w_post_p = beta_w - E_ns[m,p]
            
            log_like = -0.5*np.log(norm_sigma_n_prev_n)-(alpha_w+m/2)*np.log(beta_w_post_p)\
                    +((m-1)/2+alpha_w)*np.log(beta_w - E_ns[n,p]) \
                    +scipy.special.loggamma(m/2+alpha_w)-scipy.special.loggamma(n/2+alpha_w) # -1/2*np.log(2*np.pi)

            log_like1 = -0.5*np.log(norm_sigma_n_prev_n)
            
            # import pdb;pdb.set_trace()

            log_likes[p] = log_like

        # normalise weights
        log_marg += log_like

        # log_marg1 += log_like1

                
        # mse += (n_mus[m,0,0] - y_n)**2
        # mse+= (y_hat[0] - y_n)**2

        if m < N-1:
            pred_y[m+1] = y_hat[0]
            mse += (y_hat[0] - x_ns[m+1])**2
            hr += (np.sign((pred_y[m+1] - pred_y[m])*(x_ns[m+1]-x_ns[m]))==1)

        if m < N-step:
            pred_y3[m+step] = y_hat3
            mse3 += (y_hat3 - x_ns[m+step])**2
            hr3 += (np.sign((pred_y[m+step] - pred_y[m])*(x_ns[m+step]-x_ns[m]))==1)
        # import pdb;pdb.set_trace()

    # tot_log_marg = log_marg1 - (alpha_w+m/2)*np.log(beta_w_post_p)
    print(-E_ns[-2], beta_w_post_p, log_marg1, (alpha_w+m/2)*np.log(beta_w_post_p))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars,n_log_ws, E_ns, log_marg, mse, mse3, pred_y, hr, hr3
   

def particle_filter_2d(y_ns, P, c, T, sigma_mus, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    n_mus = np.zeros((N, P, 6))
    n_vars = np.zeros((N, P, 6, 6))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    var_0 = np.identity(6)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 10, 10)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.zeros((2,6))
    observation_matrix[0, :3] = np.array([1,0,0])
    observation_matrix[1, 3:] = np.array([1,0,0])

    log_marg = 0.0
    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            try:
                n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
            except:
                import pdb;pdb.set_trace()
        
        log_likes = np.zeros(P)
        # update
        for p in range(P):
            vs, gammas = generate_jumps(c, T, delta_t) # assume same jump for now
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t,c, T, l)
            
            # transition matrices
            C1 = noise_variance_C(processer, sigma_w, sigma_mus[0])
            C2 = noise_variance_C(processer, sigma_w, sigma_mus[1])
            A1 = transition_matrix(processer)

            C, A = np.zeros((6,6)), np.zeros((6,6))
            C[:3,:3] = C1
            C[3:,3:] = C2
            A[:3,:3] = A1
            A[3:,3:] = A1

            # C = noise_Cs[m]
            # A = trans_As[m]
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter2d(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n)
            
            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[m,p] = -(y_n-y_hat_n_prev_n).T@np.linalg.inv(norm_sigma_n_prev_n)@(y_n-y_hat_n_prev_n)/2
            beta_w_post_p = beta_w - sum(E_ns[:,p])
            log_like = -0.5*np.log(np.linalg.det(sigma_n_prev_n))-(alpha_w+m*2/2)*np.log(beta_w_post_p)\
                    +((m-1)*2/2+alpha_w)*np.log(beta_w - sum(E_ns[:m,p]))+\
                    scipy.special.loggamma(m*2/2+alpha_w)-scipy.special.loggamma(n*2/2+alpha_w) # -2/2*np.log(2*np.pi)
            n_log_ws[m,p] = n_log_ws[n, p]+ log_like

            log_likes[p] = log_like

        # normalise weights
        # log_marg += np.log(sum(np.exp(log_likes))/P)
        log_marg += np.log(np.exp(n_log_ws[m,:]).sum())
        n_log_ws[m,:] = n_log_ws[m,:]- np.log(sum(np.exp(n_log_ws[m,:])))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns, log_marg

def particle_filter_2d_w_drift(y_ns, P, c, T, sigma_mus, sigma_betas, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    D=4
    n_mus = np.zeros((N, P, D*2))
    n_vars = np.zeros((N, P, D*2, D*2))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    var_0 = np.identity(D*2)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 10, 10)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.zeros((2,D*2))
    observation_matrix[0, :D] = np.array([1,0,0,0])
    observation_matrix[1, D:] = np.array([1,0,0,0])

    log_marg = 0.0
    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            try:
                n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
            except:
                import pdb;pdb.set_trace()
        
        log_likes = np.zeros(P)
        # update
        for p in range(P):
            vs, gammas = generate_jumps(c, T, delta_t) # assume same jump for now
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t,c, T, l)
            
            # transition matrices
            C1 = noise_variance_C_w_drift(processer, sigma_w, sigma_mus[0], sigma_betas[0])
            C2 = noise_variance_C_w_drift(processer, sigma_w, sigma_mus[1], sigma_betas[1])
            A1 = transition_matrix_w_drift(processer)

            C, A = np.zeros((2*D,2*D)), np.zeros((2*D,2*D))
            C[:D,:D] = C1
            C[D:,D:] = C2
            A[:D,:D] = A1
            A[D:,D:] = A1

            # C = noise_Cs[m]
            # A = trans_As[m]
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter2d(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n)
            
            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[m,p] = -(y_n-y_hat_n_prev_n).T@np.linalg.inv(norm_sigma_n_prev_n)@(y_n-y_hat_n_prev_n)/2
            beta_w_post_p = beta_w - sum(E_ns[:,p])
            log_like = -0.5*np.log(np.linalg.det(sigma_n_prev_n))-(alpha_w+m*2/2)*np.log(beta_w_post_p)\
                    +((m-1)*2/2+alpha_w)*np.log(beta_w - sum(E_ns[:m,p]))+\
                    scipy.special.loggamma(m*2/2+alpha_w)-scipy.special.loggamma(n*2/2+alpha_w) # -2/2*np.log(2*np.pi)
            n_log_ws[m,p] = n_log_ws[n, p]+ log_like

            log_likes[p] = log_like

        # normalise weights
        log_marg += np.log(sum(np.exp(log_likes))/P)
        n_log_ws[m,:] = n_log_ws[m,:]- np.log(sum(np.exp(n_log_ws[m,:])))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns, log_marg

def particle_filter_1d_w_drift(y_ns, P, c, T, sigma_mu, sigma_beta, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    D=4
    n_mus = np.zeros((N, P, D))
    n_vars = np.zeros((N, P, D, D))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    var_0 = np.identity(D)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 10, 10)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.array([1,0,0,0])

    log_marg = 0.0
    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            try:
                n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
            except:
                import pdb;pdb.set_trace()
        
        log_likes = np.zeros(P)
        # update
        for p in range(P):
            vs, gammas = generate_jumps(c, T, delta_t) # assume same jump for now
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t,c, T, l)
            
            # transition matrices
            C = noise_variance_C_w_drift(processer, sigma_w, sigma_mu, sigma_beta)
            A = transition_matrix_w_drift(processer)

            # C = noise_Cs[m]
            # A = trans_As[m]
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n)
            
            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[m,p] = -(y_n-y_hat_n_prev_n)**2/(norm_sigma_n_prev_n)/2
            beta_w_post_p = beta_w - sum(E_ns[:,p])
            log_like = -0.5*np.log(norm_sigma_n_prev_n)-(alpha_w+m/2)*np.log(beta_w_post_p)\
                    +((m-1)/2+alpha_w)*np.log(beta_w - sum(E_ns[:m,p]))+\
                    scipy.special.loggamma(m/2+alpha_w)-scipy.special.loggamma(n/2+alpha_w)# -2/2*np.log(2*np.pi)
            n_log_ws[m,p] = n_log_ws[n, p]+ log_like

            log_likes[p] = log_like

        # normalise weights
        log_marg += np.log(sum(np.exp(log_likes))/P)
        n_log_ws[m,:] = n_log_ws[m,:]- np.log(sum(np.exp(n_log_ws[m,:])))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns, log_marg

def particle_filter_3d(y_ns, P, c, T, sigma_mus, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    n_mus = np.zeros((N, P, 9))
    n_vars = np.zeros((N, P, 9, 9))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))
    dim = 3

    # initialize x
    var_0 = np.identity(9)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 10, 10)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.zeros((3,9))
    observation_matrix[0, :3] = np.array([1,0,0])
    observation_matrix[1, 3:6] = np.array([1,0,0])
    observation_matrix[2, 6:] = np.array([1,0,0])

    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            try:
                n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
            except:
                import pdb;pdb.set_trace()
        
        # update
        for p in range(P):
            vs, gammas = generate_jumps(c, T, delta_t) # assume same jump for now
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t,c, T, l)
            
            # transition matrices
            C1 = noise_variance_C(processer, sigma_w, sigma_mus[0])
            C2 = noise_variance_C(processer, sigma_w, sigma_mus[1])
            C3 = noise_variance_C(processer, sigma_w, sigma_mus[2])
            A1 = transition_matrix(processer)

            C, A = np.zeros((9,9)), np.zeros((9,9))
            C[:3,:3] = C1
            C[3:6,3:6] = C2
            C[6:,6:] = C3
            A[:3,:3] = A1
            A[3:6,3:6] = A1
            A[6:,6:] = A1

            # C = noise_Cs[m]
            # A = trans_As[m]
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter2d(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n,dim=3)
            
            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[m,p] = -(y_n-y_hat_n_prev_n).T@np.linalg.inv(norm_sigma_n_prev_n)@(y_n-y_hat_n_prev_n)/2
            beta_w_post_p = beta_w - sum(E_ns[:,p])
            log_like = -0.5*np.log(np.linalg.det(sigma_n_prev_n))-(alpha_w+m*dim/2)*np.log(beta_w_post_p)\
                    +((m-1)*dim/2+alpha_w)*np.log(beta_w - sum(E_ns[:m,p]))+\
                    scipy.special.loggamma(m*dim/2+alpha_w)-scipy.special.loggamma(n*dim/2+alpha_w) # -2/2*np.log(2*np.pi)
            n_log_ws[m,p] = n_log_ws[n, p]+ log_like

        # normalise weights
        n_log_ws[m,:] = n_log_ws[m,:]- np.log(sum(np.exp(n_log_ws[m,:])))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns

def process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w, sigmaw_range=[0.8,1.3],alpha_w = 0.000000000001,beta_w = 0.000000000001):
    # examine data
    (N, num_particles, D) = n_mus.shape
    
    average = np.zeros((N, D))
    std3 = np.zeros((N, D))
    avg_P = np.zeros((N, D, D))

    mean_sigmas = np.zeros((N,num_particles))
    var_sigmas = np.zeros((N,num_particles))
    betas = np.zeros((N, num_particles))
    for n in range(N):
        alpha_n = alpha_w + (n+1)
        for p in range(num_particles):
            # betas[n,p] = beta_w - sum(E_ns[:(n+2),p])
            betas[n,p] = beta_w - E_ns[(n+1),p]
            mean_sigmas[n,p] = betas[n,p]/(alpha_n-1)
            var_sigmas[n,p] = max(0, betas[n,p]**2/(alpha_n-1)**2/(alpha_n-2))

    tot_mean_sigma = np.zeros(N)
    tot_var_sigma = np.zeros(N)
    
    if D == 3: # test if it's 2 dim
        d_a = 1
    elif D == 6 or D == 8:
        d_a = 2
    else:
        d_a = 3

    for i in range(N):
        alpha_n = alpha_w + i*d_a/2
        for d in range(D):
            average[i,d]=np.dot(n_mus[i,:,d], np.exp(n_log_ws[i,:]))
            for j in range(D):
                if i<=2:
                    var_x_ij = n_vars[i,:,d,j]*sigma_w**2
                else:
                    var_x_ij = n_vars[i,:,d,j]*betas[i,:]/(alpha_n-d_a/2)
                avg_P[i,d,j] = np.dot(var_x_ij, np.exp(n_log_ws[i,:]))+\
                np.dot((n_mus[i,:,d]-average[i,d])*(n_mus[i,:,j]-average[i,j]), np.exp(n_log_ws[i,:]))
                
            std3[i,d]=np.sqrt(avg_P[i,d,d])*3

        tot_mean_sigma[i]=np.dot(mean_sigmas[i,:], np.exp(n_log_ws[i,:]))
        tot_var_sigma[i] = np.dot(var_sigmas[i,:], np.exp(n_log_ws[i,:])) + np.dot((mean_sigmas[i,:]-tot_mean_sigma[i])*(mean_sigmas[i,:]-tot_mean_sigma[i]),np.exp(n_log_ws[i,:]))
    
    alpha = alpha_w + N*d_a/2
    xs = np.linspace(sigma_w**2*sigmaw_range[0],sigma_w**2*sigmaw_range[1],100)
    fxs = 0
    for p in range(num_particles):
        x_beta = xs/betas[-1,p]
        pdf = invgamma.pdf(x_beta, alpha)/betas[-1,p]*np.exp(n_log_ws[-1,p])
        fxs += pdf

    return average, std3, betas, xs, fxs

def inf_1d_fish(num_particles = 200, N=1000, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt',
                l = -1e-2, c=5, noise_sig=0.1,delta_t=1, alpha=0.9, k_v=500, dim=0, sigma_mu=1e-2, sigma_w=1,
                startx = 3000,sigma_mu0 = 0.0,m=400):

    x_ns = extract_track(datapath)[0,startx:(startx+N),dim]
    # add noise
    y_ns = x_ns  +np.random.normal(0, noise_sig, N)
    # y_ns = x_ns #+np.random.normal(0, sigma_w*k_v, N)

    n_mus, n_vars, n_log_ws, E_ns, _, _,_,_,_ = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_t, sigma_mu0)
    average, std3, _ ,_, _ = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    # with open('experiments\figure\simplified\fish\1d_marginals.txt', 'w') as f:
    #     line = f'dim={dim}, l={l}, c={c}, N={N}, T={delta_t}, sigma w ={sigma_w}, sigma mu={sigma_mu}, alpha = {alpha}, kv={k_v} \nmarginals:{marg}\n'
    #     f.write(line)
    # np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish1d_x",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)
    pred_xs = average[m:,:2]
    y_ns = y_ns[m:]
    x_ns = x_ns[m:]
    N = N - m

    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    plt.ylabel('displacement')
    plt.plot(range(len(y_ns)),pred_xs[:,0], label='pred')
    plt.plot(range(len(y_ns)),x_ns, label='true')
    plt.scatter(range(len(y_ns)), y_ns, label='noisy',s=5, color='pink')

    plt.ylim([min(average[m:,0] - std3[m:,0]),max(average[m:,0] + std3[m:,0])])
    plt.fill_between(range(N), average[m:,0] - std3[m:,0], average[m:,0] + std3[m:,0],
                 color='gray', alpha=0.2)
    
    # plt.scatter(range(N),y_ns,color='orange',s=5)
    plt.legend(['pred','noisy'])

    plt.subplot(3,1,2)
    plt.ylabel('velocity')
    plt.plot(pred_xs[:,1])
    plt.hlines([0.0], 0, N,linestyle = '--', color = 'green')
    # plt.plot(x_ns, linestyle = '--', color = 'red')

    # plt.ylim([min(average[200:,1] - std3[200:,1]),max(average[200:,1] + std3[200:,1])])
    # # plt.ylim([-0.5,0.5])
    # plt.fill_between(range(len(average)), average[:,1] - std3[:,1], average[:,1] + std3[:,1],
                #  color='gray', alpha=0.2)

    plt.subplot(3,1,3)
    plt.ylabel('mu')
    plt.plot(average[m:,-1])
    plt.hlines([0.0], 0, N,linestyle = '--', color = 'green')
    plt.ylim([min(average[m:,-1] - std3[m:,-1]),max(average[m:,-1] + std3[m:,-1])])
    plt.fill_between(range(N), average[m:,-1] - std3[m:,-1], average[m:,-1] + std3[m:,-1],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/fish1d/xs_{int(alpha*10)}_l{int(abs(l))}.png')

def inf_finance(num_particles = 200, N=500,N0=0, data = 'nvdia', alpha = 0.8, l = -1, c=10, sigma_w = 2, sigma_mu = 1, k_v=1, returnlmarg=True, step=5, noise_sig=0.1,K=16 ):
    if data == 'finance':
        # c = 1e5
        datapath = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\dataEurUS.mat"
        x_ns, t_ns = extract_mat_data(datapath)
        start_time = 7.3259e5
        end_time = start_time + 0.1
        # import pdb;pdb.set_trace()
        mask = (t_ns>start_time) & (t_ns<end_time)
        x_ns = x_ns[mask]
        x_ns -= x_ns[0]
        t_ns =( t_ns[mask] - start_time)*1e3 #*1e4
        delta_ts = t_ns[1:] - t_ns[:-1]
        
        # delta_ts = 1
    elif data == 'fish1d':
        datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'
        startx = 3000+N0
        tracks = extract_track(datapath)
        dims=0
        x_ns = tracks[0,startx:(startx+N),dims].T
        x_ns -= x_ns[0]
        delta_ts = 1
        t_ns = np.linspace(0, len(x_ns),endpoint=False,num=len(x_ns))
    elif data == 'nvdia_tl':
        csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVDA.USUSD_Ticks_20.06.2025-20.06.2025.csv"
        nvda_data_reloaded= pd.read_csv(csvfile)
        x_ns = (np.array(nvda_data_reloaded['Ask'])+np.array(nvda_data_reloaded['Bid']))/2

        nvda_data_reloaded['Local time'] = pd.to_datetime(nvda_data_reloaded['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
        nvda_data_reloaded['Local time'] = (nvda_data_reloaded['Local time'] - nvda_data_reloaded['Local time'].iloc[0]).dt.total_seconds()

        t_ns = np.array(nvda_data_reloaded['Local time'])

        x_ns, t_ns = x_ns[N0:N0+N]-x_ns[N0], t_ns[N0:N0+N]
        delta_ts = t_ns[1:] - t_ns[:-1]
    else:
        csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"
        nvda_data_reloaded= pd.read_csv(csvfile, index_col=0, parse_dates=True)
        x_ns = np.array(nvda_data_reloaded['Price']['2025-01-29'])
        x_ns = (x_ns - x_ns[0])
        x_ns = np.flip(x_ns) # pick the prev 300 data for const mu
        N=len(x_ns)
        delta_ts = 1
        t_ns = np.linspace(0, len(x_ns),endpoint=False,num=len(x_ns))

    T = 1

    # add noise
    if data == 'fish1d':
        rng = default_rng(42)
        b = rng.normal(0.0, noise_sig, N)
        y_ns = x_ns  + b
    else:
        y_ns = x_ns # +np.random.normal(0, sigma_w*k_v, N)
    # plt.plot(t_ns, x_ns)
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # plt.savefig(r'experiments\figure\real_data\finance\noisy')


    n_mus, n_vars, n_log_ws, E_ns, log_marg, mse,mse1,  y_1step, y_nstep, hr, hr1 = particle_filter_1d(y_ns,x_ns, num_particles, c, T, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_ts, K=K, step=step)
    # n_mus, n_vars, n_log_ws, E_ns, log_marg = particle_filter_1d_w_drift(y_ns, num_particles, c, T, sigma_mu, sigma_beta,sigma_w, k_v*sigma_w, alpha, l, delta_ts)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)

    if returnlmarg:
        return log_marg, mse/(len(y_ns)-step), mse1/(len(y_ns)-1), np.mean((average[:,0]-x_ns)**2)
    else:
        print(f'log marg: {log_marg}, mse: {mse/(len(y_ns)-step)}, mse1: {mse1/(len(y_ns)-1)}, mse0:{np.mean((average[:,0]-x_ns)**2)}, ')
        print(f'hit rate: {hr/(len(y_ns)-step)}, hit rate 1: {hr1/(len(y_ns)-step)}')

    
    # with open(f'experiments/figure/wdrift/{data}/marginals.txt', 'a') as f:
    #     line = f'l={l}, c={c}, N={N}, dt={delta_ts}, sigma w ={sigma_w}, sigma mu={sigma_mu}, alpha = {alpha}, kv={k_v} \nlog marginals:{log_marg}\n'
    #     f.write(line)
    # np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\finance",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, marg = marg, allow_pickle=True)

    plt.figure(figsize=(8,5))
    if data == 'nvdia_tl':
        plt.ylabel('Stock Price')
        plt.xlabel('Seconds')
    elif data == 'fish1d':
        plt.ylabel('Displacement')
        plt.xlabel('n')
    pred_xs = average[:,:2]
    # plt.plot(t_ns, pred_xs[:,0])
    plt.plot(t_ns, y_1step, label='1 step pred')
    if data == 'fish1d':
        plt.plot(t_ns, x_ns, label = 'True')
        plt.scatter(t_ns, y_ns, color='pink',s=5, label='Noisy Obs')
    else:
        plt.scatter(t_ns, x_ns,color='red',s=5, label='Noisy Obs')
    # plt.plot(t_ns, y_nstep- x_ns, linestyle = '--', label = '3 step pred')
    if data == 'nvdia_tl':
        miny = -0.3
        plt.ylim([miny,0.09])
        # plt.ylim([-4.5,4])
        plt.fill_between(t_ns, miny*np.ones_like(t_ns), miny+np.abs(y_1step- x_ns),label='Error',
                    color='gray', alpha=0.2)
    plt.legend()
    plt.savefig(f'experiments/figure/simplified/{data}/xs_{int(alpha*10)}_l{int(abs(l))}_1err.png')
    plt.show()

    b=25
    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    if data == 'nvdia_tl':
        plt.ylabel('Stock Price')
    elif data == 'fish1d':
        plt.ylabel('Displacement')
    
    pred_xs = average[:,:2]
    plt.plot(t_ns, pred_xs[:,0], label='Particle Mean')
    # plt.plot(t_ns, x_ns, linestyle = '--', color = 'red')
    if data == 'fish1d':
        plt.plot(t_ns, x_ns, label = 'True')
        plt.scatter(t_ns, y_ns, color='pink',s=5, label='Noisy Obs')
    else:
        plt.scatter(t_ns, x_ns,color='pink',s=5, label='Noisy Obs')
    # plt.plot(t_ns, y_1step, linestyle = '--', color = 'red')
    plt.ylim([min(average[b:,0] - std3[b:,0]),max(average[b:,0] + std3[b:,0])])
    plt.fill_between(t_ns, average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.ylabel('velocity')
    
    # plt.ylim([-1,1])
    plt.plot(t_ns, pred_xs[:,1])
    # plt.plot(t_ns, n_mus[:,:,1], linewidth = 0.5)
    plt.hlines([0.0], t_ns[0], t_ns[-1],linestyle = '--', color = 'green')
    if data == 'nvdia_tl':
        plt.xlabel('Seconds')
        # plt.ylim([min(average[25:,1] ),max(average[25:,1] )])
        plt.ylim([-0.1,0.1])
    elif data == 'fish1d':
        plt.xlabel('n')
        plt.ylim([-0.3,0.3])
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # plt.ylim([min(average[25:,1] ),max(average[25:,1] )])
    # plt.ylim([-0.5,0.5])
    # plt.fill_between(t_ns, average[:,1] - std3[:,1], average[:,1] + std3[:,1],
    #              color='gray', alpha=0.2)
    
    
    plt.subplot(3,1,3)
    plt.ylabel(r'$\mu$')
    if data == 'nvdia_tl':
        plt.xlabel('Seconds')
        plt.ylim([min(average[200:,-1] - std3[200:,-1]),max(average[200:,-1] + std3[200:,-1])])
    elif data == 'fish1d':
        plt.xlabel('n')
        plt.ylim([-0.0002,0.0002])
    plt.plot(t_ns, average[:,-1])
    plt.hlines([0.0], t_ns[0], t_ns[-1],linestyle = '--', color = 'green')
    # plt.ylim([min(average[200:,-1] - std3[200:,-1]),max(average[200:,-1] + std3[200:,-1])])
    
    # plt.ylim([-np.mean(np.abs(average[25:,-1]))*3,np.mean(np.abs(average[25:,-1]))*3])
    # plt.fill_between(t_ns, average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                #  color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/{data}/xvs_{int(alpha*10)}_l{int(abs(l))}.png')
    plt.show()
    return
    plt.figure()
    plt.plot(xs, fxs)
    # plt.axvline(x = sigma_w**2, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2$')
    plt.ylabel('Posterior')
    plt.savefig(f'experiments/figure/simplified/{data}/sigma_{int(alpha*10)}.png')
    plt.show()

def inf_finance_w_drift(num_particles = 200, data = 'nvdia', alpha = 0.8, l = -1, c=10, sigma_w = 2, sigma_mu = 1, sigma_beta = 1, k_v=1):
    if data == 'finance':
        N = 500
        datapath = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\dataEurUS.mat"
        x_ns, t_ns = extract_mat_data(datapath)
        x_ns = x_ns[:N]*5e4 - x_ns[0]*5e4
        t_ns = t_ns[:N]*5e4
        delta_ts = t_ns[1:] - t_ns[:-1]
    else:
        csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"
        nvda_data_reloaded= pd.read_csv(csvfile, index_col=0, parse_dates=True)
        x_ns = np.array(nvda_data_reloaded['Price']['2025-01-29'])
        x_ns = (x_ns - x_ns[0])
        x_ns = np.flip(x_ns)[:300]
        N=len(x_ns)
        delta_ts = 1
        t_ns = np.linspace(0, len(x_ns),endpoint=False,num=len(x_ns))

    T = 1

    # add noise
    y_ns = x_ns # +np.random.normal(0, sigma_w*k_v, N)

    n_mus, n_vars, n_log_ws, E_ns, log_marg = particle_filter_1d_w_drift(y_ns, num_particles, c, T, sigma_mu, sigma_beta,sigma_w, k_v*sigma_w, alpha, l, delta_ts)
    # return log_marg

    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    # with open(f'experiments/figure/wdrift/{data}/marginals.txt', 'a') as f:
    #     line = f'l={l}, c={c}, N={N}, dt={delta_ts}, sigma w ={sigma_w}, sigma mu={sigma_mu}, alpha = {alpha}, kv={k_v} \nlog marginals:{log_marg}\n'
    #     f.write(line)
    # np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\finance",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, marg = marg, allow_pickle=True)

    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    plt.ylabel('Centered Stock Price')
    plt.xlabel('Minutes')
    pred_xs = average[:,:2]
    plt.plot(t_ns, pred_xs[:,0])
    # plt.plot(t_ns, x_ns, linestyle = '--', color = 'red')
    plt.scatter(t_ns, x_ns,color='pink',s=5)
    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(t_ns, average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    plt.legend(['Particle mean','Data'])
    
    # plt.subplot(3,1,2)
    # plt.ylabel('velocity')
    # plt.plot(t_ns, pred_xs[:,1])
    # # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # # plt.ylim([min(average[25:,1] - std3[25:,1]),max(average[25:,1] + std3[25:,1])])
    # plt.ylim([-0.5,0.5])
    # plt.fill_between(t_ns, average[:,1] - std3[:,1], average[:,1] + std3[:,1],
                #  color='gray', alpha=0.2)
    
    
    plt.subplot(3,1,2)
    plt.ylabel('mu')
    plt.plot(t_ns, average[:,-2])
    plt.hlines([0.0], t_ns[0], t_ns[-1],linestyle = '--', color = 'green')
    plt.ylim([min(average[25:,-2] - std3[25:,-2]),max(average[25:,-2] + std3[25:,-2])])
    # plt.ylim([-0.02,0.02])
    # plt.ylim([-np.mean(np.abs(average[25:,-1]))*3,np.mean(np.abs(average[25:,-1]))*3])
    plt.fill_between(t_ns, average[:,-2] - std3[:,-2], average[:,-2] + std3[:,-2],
                 color='gray', alpha=0.2)
    
    plt.subplot(3,1,3)
    plt.ylabel('beta')
    plt.plot(t_ns, average[:,-1])
    plt.hlines([0.0], t_ns[0], t_ns[-1],linestyle = '--', color = 'green')
    # plt.ylim([min(average[25:,-1] - std3[25:,-1]),max(average[25:,-1] + std3[25:,-1])])
    plt.ylim([-0.2,0.2])
    # plt.ylim([-np.mean(np.abs(average[25:,-1]))*3,np.mean(np.abs(average[25:,-1]))*3])
    plt.fill_between(t_ns, average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/wdrift/{data}/xs_{int(alpha*10)}_l{int(abs(l))}.png')
    # return
    plt.figure()
    plt.plot(xs, fxs)
    # plt.axvline(x = sigma_w**2, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2$')
    plt.ylabel('Posterior')
    plt.savefig(f'experiments/figure/wdrift/{data}/sigma_{int(alpha*10)}.png')

def inf_2d_fish(num_particles = 100, N=1000, m=200,noise_sig=0.1, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt', k_v = 1e2, sigma_w = 0.1,\
                sigma_mu = 1e-6, alpha = 1.2, l = -1e-2):
    
    c = 10
    startx = 3000
    delta_t = 1
    sigma_mus = np.ones(2)*sigma_mu
    
     # 1.5e4 for alpha=0.9

    tracks = extract_track(datapath)
    dims=[0,2]

    x_ns = tracks[0,startx:(startx+N),dims].T 
    y_ns = x_ns  +np.random.multivariate_normal([0,0], noise_sig*np.eye(2), N)

    n_mus, n_vars, n_log_ws, E_ns, _ = particle_filter_2d(y_ns, num_particles, c, delta_t, sigma_mus, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,_, _ = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish2d",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)
    # with open('experiments\figure\simplified\fish\marginals.txt', 'w') as f:
    #     line = f'dim={dims}, l={l}, c={c}, N={N}, T={delta_t}, sigma w ={sigma_w}, sigma mu={sigma_mus}, alpha = {alpha}, kv={k_v}, \nmarginals:{marg}\n'
    #     f.write(line)

    pred_xs = average[m:,:]
    y_ns = y_ns[m:,:]
    x_ns = x_ns[m:,:]
    N = N - m
    plt.ylabel('displacement')
    plt.plot(pred_xs[:,0], pred_xs[:,3], label='Pred')
    plt.scatter(y_ns[:,0], y_ns[:,1], linestyle = '--', color = 'pink',s=5, label='Noisy')
    plt.plot(x_ns[:,0], x_ns[:,1], label = 'True')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    plt.quiver(pred_xs[quiver_idx,0], pred_xs[quiver_idx,3], pred_xs[quiver_idx,2], pred_xs[quiver_idx,-1], label='Mu', zorder=10)
    plt.plot(x_ns[0,0], x_ns[0,1], 'go', label='Start')
    plt.legend()
    plt.savefig(f'experiments/figure/simplified/fish/2d_xs_{int(alpha*10)}_l{int(abs(l))}.png')

    return 
    # animation
    fig, ax = plt.subplots()
    line1 = ax.plot(pred_xs[0,0], pred_xs[0,3], label='Pred')[0]
    line2 = ax.plot(y_ns[0,0], y_ns[0,1], linestyle = '--', color = 'red', label='Noisy')[0]
    scale = 5
    vecfield = ax.quiver(pred_xs[0,0], pred_xs[0,3], pred_xs[0,2]*scale, pred_xs[0,-1]*scale, label='Mu')
    ax.set(xlim=[min(pred_xs[:,0]) - 2.5, max(pred_xs[:,0]) + 2.5], \
           ylim=[min(pred_xs[:,3]) - 2.5, max(pred_xs[:,3]) + 2.5])

    def update(frame):
        # update the line plot:
        line1.set_xdata(pred_xs[:frame,0])
        line1.set_ydata(pred_xs[:frame,3])
        line2.set_xdata(y_ns[:frame,0])
        line2.set_ydata(y_ns[:frame,1])
        vecfield.set_offsets([pred_xs[frame,0], pred_xs[frame,3]])
        vecfield.set_UVC(pred_xs[frame,2]*scale, pred_xs[frame,-1]*scale)
        return (line1, line2, vecfield)
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=30)
    plt.show()
    # return
    ani.save(filename=f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/alpha_stable_levy/stable_levy_code/data/real_data/video/2d_xs_{int(alpha*10)}_l{int(abs(l))}.gif', writer='pillow')

    plt.subplot(2,1,2)
    plt.ylabel('velocity')
    plt.plot(pred_xs[:,1], pred_xs[:,4], label = 'Pred')
    plt.plot(pred_xs[0,1], pred_xs[0,4],'go', label='Start')
    plt.legend()
    




    

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('mu x')
    idx = 2
    plt.plot(pred_xs[:,idx])
    plt.ylim([min(pred_xs[:,idx] - std3[m:,idx]),max(pred_xs[:,idx] + std3[m:,idx])])
    plt.fill_between(range(len(pred_xs)), pred_xs[:,idx] - std3[m:,idx], pred_xs[:,idx] + std3[m:,idx],
                 color='gray', alpha=0.2)
    
    plt.subplot(2,1,2)
    plt.ylabel('mu z')
    idx = -1
    plt.plot(pred_xs[:,idx])
    plt.ylim([min(pred_xs[:,idx] - std3[m:,idx]),max(pred_xs[:,idx] + std3[m:,idx])])
    plt.fill_between(range(len(pred_xs)), pred_xs[:,idx] - std3[m:,idx], pred_xs[:,idx] + std3[m:,idx],
                 color='gray', alpha=0.2)
    
    plt.show()
    
def inf_3d_fish(alpha = 1.6, num_particles = 100, N=500, m =100,scale = 1e3, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -1e-2
    c = 5
    startx = 3000
    delta_t = 1
    sigma_w = 0.1
    sigma_mus = [1e-1, 1e-1, 1e-1]
    k_v = 100 # 1.5e4 for alpha=0.9

    tracks = extract_track(datapath)
    y_ns = tracks[0,startx:(startx+N),:]

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_3d(y_ns, num_particles, c, delta_t, sigma_mus, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,_, _ = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish3d",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)
    # with open('experiments/figure/simplified/fish/marginals.txt', 'w') as f:
    #     line = f'dim = 3d, l={l}, c={c}, N={N}, T={delta_t}, sigma w ={sigma_w}, sigma mu={sigma_mus}, alpha = {alpha}, kv={k_v} \nmarginals:{marg}\n'
    #     f.write(line)

    pred_xs = average[m:,:]
    y_ns = y_ns[m:,:]
    N = N - m
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pred_xs[:,0], pred_xs[:,3], pred_xs[:,6], label='Pred')
    ax.plot(y_ns[:,0], y_ns[:,1], y_ns[:,2], linestyle = '--', color = 'red', label='Noisy')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    ax.quiver(pred_xs[quiver_idx,0], pred_xs[quiver_idx,3],pred_xs[quiver_idx,6], pred_xs[quiver_idx,2]*scale,pred_xs[quiver_idx,5]*scale, pred_xs[quiver_idx,-1]*scale, label='Mu',color='black')
    # ax.plot(y_ns[0,0], y_ns[0,1],y_ns[0,2], 'bo', label='Start')
    plt.legend()
    plt.savefig(f'experiments/figure/simplified/fish/3d_xs_{int(alpha*10)}_l{int(abs(l))}.png')

    # animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    line1 = ax.plot(pred_xs[0,0], pred_xs[0,3],pred_xs[0,6], label='Pred')[0]
    line2 = ax.plot(y_ns[0,0], y_ns[0,1], y_ns[0,2],linestyle = '--', color = 'red', label='Noisy')[0]
    ax.set(xlim=[min(pred_xs[:,0]) - 2.5, max(pred_xs[:,0]) + 2.5], \
           ylim=[min(pred_xs[:,3]) - 2.5, max(pred_xs[:,3]) + 2.5], \
            zlim=[min(pred_xs[:,6]) - 2.5, max(pred_xs[:,6]) + 2.5])
    
    
    # ax2 = fig.add_subplot(projection='3d')
    vecfield = ax.quiver(pred_xs[0,0], pred_xs[0,3],pred_xs[0,6], pred_xs[0,2]*scale,pred_xs[0,5]*scale, pred_xs[0,-1]*scale, label='Mu', color='black')

    def update(frame):
        # update the line plot:
        line1.set_xdata(pred_xs[:frame,0])
        line1.set_ydata(pred_xs[:frame,3])
        line1.set_3d_properties(pred_xs[:frame,6])
        line2.set_xdata(y_ns[:frame,0])
        line2.set_ydata(y_ns[:frame,1])
        line2.set_3d_properties(y_ns[:frame,2])

        # vecfield.set_offsets([pred_xs[frame,0], pred_xs[frame,3]])
        # import pdb;pdb.set_trace()
        new_arrow = ax.quiver(pred_xs[frame,0], pred_xs[frame,3],pred_xs[frame,6], pred_xs[frame,2]*scale,pred_xs[frame,5]*scale, pred_xs[frame,-1]*scale, label='Mu', modify=True, color='black')

        vecfield.set(segments = new_arrow)
        # vecfield.set(offsets=[pred_xs[frame,0], pred_xs[frame,3]])
        # vecfield.set(segments= [[(pred_xs[frame,2]*scale, pred_xs[frame,5]*scale, pred_xs[frame, 8])] ])
        return (line1, line2, vecfield)
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=30)
    plt.legend()
    plt.show()
    # return
    ani.save(filename=f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/alpha_stable_levy/stable_levy_code/data/real_data/video/3d_xs_{int(alpha*10)}_l{int(abs(l))}.gif', writer="pillow")

def test_data_2d(num_particles = 200, alpha = 1.6, l = None):
    datapath = f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/2d/x_ns_{int(alpha*10)}.npz'
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    y_ns = data_read['y'] # observation, size (N, 1)
    if not l:
        l = data_read['l']
    c = data_read['c']
    delta_t = data_read['delta_t']
    sigma_w = data_read['sigma_w']
    sigma_mus = data_read['sigma_mu']
    alpha = data_read['alpha']
    k_v = data_read['k_v']

    n_mus, n_vars, n_log_ws, E_ns, log_marg = particle_filter_2d(y_ns, num_particles, c, delta_t, sigma_mus, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/2d/filter_res_{int(alpha*10)}.npz',n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    return log_marg

def test_data_2d_wdrift(num_particles = 200, alpha = 1.6, l = None):
    datapath = f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/w_drift/x_ns_{int(alpha*10)}.npz'
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    y_ns = data_read['y'] # observation, size (N, 1)
    if not l:
        l = data_read['l']
    c = data_read['c']
    delta_t = data_read['delta_t']
    sigma_w = data_read['sigma_w']
    sigma_mus = data_read['sigma_mu']
    alpha = data_read['alpha']
    k_v = data_read['k_v']
    sigma_betas = data_read['sigma_betas']

    n_mus, n_vars, n_log_ws, E_ns, log_marg = particle_filter_2d_w_drift(y_ns, num_particles, c, delta_t, sigma_mus,sigma_betas, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    # average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/w_drift/filter_res_{int(alpha*10)}.npz',n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    return log_marg

def inf_2d_fish_wdrift(num_particles = 100, N=1000, m=200, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -1e-2
    c = 10
    startx = 3000
    delta_t = 1
    sigma_w = 0.1
    sigma_mus = np.ones(2)*1e-2
    sigma_betas = np.ones(2)*1e-6
    alpha = 1.2
    k_v = 1e2 # 1.5e4 for alpha=0.9

    tracks = extract_track(datapath)
    dims=[0,2]
    y_ns = tracks[0,startx:(startx+N),dims].T

    n_mus, n_vars, n_log_ws, E_ns, log_marg = particle_filter_2d_w_drift(y_ns, num_particles, c, delta_t, sigma_mus, sigma_betas,sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,_, _ = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish2d_wdrift",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)
    # with open('experiments\figure\simplified\fish\marginals.txt', 'w') as f:
    #     line = f'dim={dims}, l={l}, c={c}, N={N}, T={delta_t}, sigma w ={sigma_w}, sigma mu={sigma_mus}, alpha = {alpha}, kv={k_v}, \nmarginals:{marg}\n'
    #     f.write(line)

    pred_xs = average[m:,:]
    y_ns = y_ns[m:,:]
    N = N - m
    D=4
    plt.ylabel('displacement')
    plt.plot(pred_xs[:,0], pred_xs[:,D], label='Pred')
    plt.plot(y_ns[:,0], y_ns[:,1], linestyle = '--', color = 'red', label='Noisy')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    plt.quiver(pred_xs[quiver_idx,0], pred_xs[quiver_idx,D], pred_xs[quiver_idx,2], pred_xs[quiver_idx,2+D], label='Mu', color = 'red', alpha=0.5)
    plt.quiver(pred_xs[quiver_idx,0], pred_xs[quiver_idx,D], pred_xs[quiver_idx,3], pred_xs[quiver_idx,3+D], label='Beta', color = 'orange',alpha=0.5)
    plt.plot(y_ns[0,0], y_ns[0,1], 'go', label='Start')
    plt.legend()
    plt.savefig(f'experiments/figure/wdrift/fish/2d_xs_{int(alpha*10)}_l{int(abs(l))}_constdrift.png')

    return 

def test_data(num_particles, datapath='C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/x_ns_test.npz'):
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    y_ns = data_read['y'][:,0] # observation, size (N, 1)
    l = data_read['l']
    c = data_read['c']
    delta_t = data_read['delta_t'].item()
    sigma_w = data_read['sigma_w']
    sigma_mu = data_read['sigma_mu']
    alpha = data_read['alpha']
    k_v = data_read['k_v']

    n_mus, n_vars, n_log_ws, E_ns,log_marg,mse = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez('C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/filter_res_test.npz',n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    return average, std3, x_dashed, xs, fxs,log_marg

def plot_result_from_stored1d(datapath = 'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/x_ns_test.npz', resultpath = 'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/filter_res_test.npz'):
    res = np.load(resultpath)
    
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    c = data_read['c']
    sigma_w = data_read['sigma_w']
    alpha = data_read['alpha']
    
    average, std3, _ ,xs, fxs = process_filter_results(res['n_mus'], res['n_vars'], res['n_log_ws'], res['E_ns'], sigma_w )

    
    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('x')
    plt.plot(average[:,0])
    plt.plot(x_dashed[:,0])
    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(range(len(average)), average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    plt.legend(['pred','true'])
    plt.subplot(2,1,2)
    plt.ylabel('v')
    plt.plot(average[:,1])
    plt.plot(x_dashed[:,1])
    dim = 1
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/simu_data/1d/xs_{int(alpha*10)}.png')
    
    plt.figure()
    plt.ylabel('x mu')
    plt.plot(average[:,2])
    plt.plot(x_dashed[:,2])
    plt.legend(['pred','true'])
    plt.ylim([min(average[25:,2] - std3[25:,2]),max(average[25:,2] + std3[25:,2])])
    plt.fill_between(range(len(average)), average[:,2] - std3[:,2], average[:,2] + std3[:,2],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/simu_data/1d/mus_{int(alpha*10)}.png')
    
    plt.figure()
    plt.plot(xs, fxs)
    plt.axvline(x = sigma_w**2, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2 / 10^{-3}$')
    plt.legend(['Posterior','True $\sigma_W^2$'])
    plt.savefig(f'experiments/figure/simplified/simu_data/1d/sigma_{int(alpha*10)}.png')
    # plt.show()

def plot_result_from_stored(alpha = 1.6):
    datapath = f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/2d/x_ns_{int(alpha*10)}.npz'
    resultpath = f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/2d/filter_res_{int(alpha*10)}.npz'
    res = np.load(resultpath)
    
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    c = data_read['c']
    sigma_w = data_read['sigma_w']
    alpha = data_read['alpha']
    y = data_read['y']
    
    average, std3, _ ,xs, fxs = process_filter_results(res['n_mus'], res['n_vars'], res['n_log_ws'], res['E_ns'], sigma_w )

    
    plt.figure()
    plt.ylabel('x')
    plt.plot(average[:,0], average[:,3])
    plt.plot(x_dashed[:,0], x_dashed[:,3])
    plt.scatter(y[:,0], y[:,1], color='pink',s=5)
    # plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    # plt.fill_between(range(len(average)), average[:,0] - std3[:,0], average[:,0] + std3[:,0],
    #              color='gray', alpha=0.2)
    plt.legend(['pred','true', 'noisy'])
    # plt.subplot(2,1,2)
    # plt.ylabel('y')
    # plt.plot(average[:,3])
    # plt.plot(x_dashed[:,3])
    # plt.ylim([min(average[25:,3] - std3[25:,3]),max(average[25:,3] + std3[25:,3])])
    # plt.fill_between(range(len(average)), average[:,3] - std3[:,3], average[:,3] + std3[:,3],
                #  color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/simu_data/2d/xs_{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('x')
    plt.plot(average[:,1])
    plt.plot(x_dashed[:,1])
    dim = 1
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.legend(['pred','true'])
    plt.subplot(2,1,2)
    plt.ylabel('y')
    plt.plot(average[:,4])
    plt.plot(x_dashed[:,4])
    dim = 4
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/simu_data/2d/vs_{int(alpha*10)}.png')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('x mu')
    plt.plot(average[:,2])
    plt.plot(x_dashed[:,2])
    plt.legend(['pred','true'])
    plt.ylim([min(average[25:,2] - std3[25:,2]),max(average[25:,2] + std3[25:,2])])
    plt.fill_between(range(len(average)), average[:,2] - std3[:,2], average[:,2] + std3[:,2],
                 color='gray', alpha=0.2)
    plt.subplot(2,1,2)
    plt.ylabel('y mu')
    plt.ylim([min(average[25:,-1] - std3[25:,-1]),max(average[25:,-1] + std3[25:,-1])])
    plt.fill_between(range(len(average)), average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.plot(average[:,-1])
    plt.plot(x_dashed[:,-1])
    plt.savefig(f'experiments/figure/simplified/simu_data/2d/mus_{int(alpha*10)}.png')
    
    plt.figure()
    plt.plot(xs, fxs)
    plt.axvline(x = sigma_w**2, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2$')
    plt.legend(['Posterior','True $\sigma_W^2$'])
    plt.savefig(f'experiments/figure/simplified/simu_data/2d/sigma_{int(alpha*10)}.png')
    # plt.show()

def plot_result_from_stored_wdrift(alpha = 1.6):
    datapath = f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/w_drift/x_ns_{int(alpha*10)}.npz'
    resultpath = f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/simu/data/w_drift/filter_res_{int(alpha*10)}.npz'
    res = np.load(resultpath)
    
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    c = data_read['c']
    sigma_w = data_read['sigma_w']
    alpha = data_read['alpha']
    y = data_read['y']

    N = len(x_dashed)
    
    average, std3, _ ,xs, fxs = process_filter_results(res['n_mus'], res['n_vars'], res['n_log_ws'], res['E_ns'], sigma_w )

    D = 4
    # plt.figure()
    # plt.ylabel('displacement')
    # plt.plot(x_dashed[:,0], x_dashed[:,D], label='X(t)')
    # quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    # plt.quiver(x_dashed[quiver_idx,0], x_dashed[quiver_idx,D], x_dashed[quiver_idx,2], x_dashed[quiver_idx,D+2], label=r'$\mu(t)$', color = 'red',alpha=0.3)
    # plt.quiver(x_dashed[quiver_idx,0], x_dashed[quiver_idx,D], x_dashed[quiver_idx,3], x_dashed[quiver_idx,3+D], label=r'$\beta(t)$', color = 'blue',alpha=0.3)
    # plt.plot(x_dashed[0,0], x_dashed[0,1], 'go', label='Start')
    # plt.legend()
    # plt.savefig(f'experiments/figure/wdrift/simulation/2d_x_mu_{int(alpha*10)}.png')

    plt.figure()
    plt.ylabel('x')
    plt.plot(average[:,0], average[:,D])
    plt.plot(x_dashed[:,0], x_dashed[:,D])
    plt.scatter(y[:,0], y[:,1], color='pink',s=5)
    # plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    # plt.fill_between(range(len(average)), average[:,0] - std3[:,0], average[:,0] + std3[:,0],
    #              color='gray', alpha=0.2)
    plt.legend(['pred','true', 'noisy'])
    # plt.subplot(2,1,2)
    # plt.ylabel('y')
    # plt.plot(average[:,3])
    # plt.plot(x_dashed[:,3])
    # plt.ylim([min(average[25:,3] - std3[25:,3]),max(average[25:,3] + std3[25:,3])])
    # plt.fill_between(range(len(average)), average[:,3] - std3[:,3], average[:,3] + std3[:,3],
                #  color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/wdrift/filter_const_drift/xs_{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('x')
    plt.plot(average[:,1])
    plt.plot(x_dashed[:,1])
    dim = 1
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.legend(['pred','true'])
    plt.subplot(2,1,2)
    plt.ylabel('y')
    dim = 1+D
    plt.plot(average[:,dim])
    plt.plot(x_dashed[:,dim])
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/wdrift/filter_const_drift/vs_{int(alpha*10)}.png')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('x mu')
    dim = 2
    plt.plot(average[:,dim])
    plt.plot(x_dashed[:,dim])
    plt.legend(['pred','true'])
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.subplot(2,1,2)
    plt.ylabel('y mu')
    dim = 2+D
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.plot(average[:,dim])
    plt.plot(x_dashed[:,dim])
    plt.savefig(f'experiments/figure/wdrift/filter_const_drift/mus_{int(alpha*10)}.png')

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('x beta')
    dim = 3
    plt.plot(average[:,dim])
    plt.plot(x_dashed[:,dim])
    plt.legend(['pred','true'])
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.subplot(2,1,2)
    plt.ylabel('y beta')
    dim = 3+D
    plt.ylim([min(average[25:,dim] - std3[25:,dim]),max(average[25:,dim] + std3[25:,dim])])
    plt.fill_between(range(len(average)), average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                 color='gray', alpha=0.2)
    plt.plot(average[:,dim])
    plt.plot(x_dashed[:,dim])
    plt.savefig(f'experiments/figure/wdrift/filter_const_drift/betas_{int(alpha*10)}.png')
    
    plt.figure()
    plt.plot(xs, fxs)
    plt.axvline(x = sigma_w**2, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2$')
    plt.legend(['Posterior','True $\sigma_W^2$'])
    plt.savefig(f'experiments/figure/wdrift/filter_const_drift/sigma_{int(alpha*10)}.png')
    # plt.show()

def marg_wrt_l(alpha = 0.9, label = 'simu_data', num_particles = 200, k =5,  save = True):
    """
    alpha = 0.9/1.2
    """
    if save:
        log_lds = np.linspace(start=-2, stop=0, num=11)
        # log_lds = np.linspace(start=-4, stop=0, num=5)
        lds = 10**log_lds
        print(lds)
        # lds = np.array([1,0.5,0.1,0.01])
        means = np.zeros(len(lds))
        stds = np.zeros(len(lds))
        for j, ld in enumerate(lds):
            margs = np.zeros(k)
            for i in range(k):
                if label == 'simu_data':
                    marg = test_data_2d(num_particles=num_particles, alpha=0.9, l=-ld)
                else:
                    marg = inf_finance(num_particles=num_particles, k_v=1e3, l=-ld, sigma_mu=1e-4, sigma_w=3.0e-4, c=5)
                margs[i] = marg
            means[j]=margs.mean()
            stds[j]=margs.std(ddof=1)

        print(means)
        np.savez(f"experiments/figure/simplified/marg/{label}_marg{int(alpha*10)}.npz",ls=log_lds, means = means, stds = stds, allow_pickle=True)

    else:
        resultpath = f"experiments/figure/simplified/marg/{label}_marg{int(alpha*10)}.npz"
        res = np.load(resultpath)
        means = res['means']
        stds = res['stds']
        log_lds = res['ls']
        lds = 10**log_lds


    plt.figure()
    plt.plot(log_lds,means)
    if label == 'simu_data':
        plt.axvline(x = np.log10(0.05), color = 'g', label = 'lambda',linestyle='dashed',)
    plt.fill_between(log_lds, means-3*stds, means+3*stds,color='gray', alpha=0.2)
    plt.xlabel(r'$\lambda$')
    plt.xticks(ticks= log_lds,labels=[round(l, 4) for l in lds])
    plt.ylabel('log marginal')
    if label == 'simu_data':
        plt.legend(['Marginals', r'True $\lambda$'])
    plt.savefig(f'experiments/figure/simplified/marg/marg_l_a{int(alpha*10)}_{label}.png')

def dist_nvidia():
    csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"
    nvda_data_reloaded= pd.read_csv(csvfile, index_col=0, parse_dates=True)
    x_ns = np.array(nvda_data_reloaded['Price']['2025-01-29'])
    x_ns = (x_ns - x_ns[0])
    x_ns = np.flip(x_ns)[:301]

    dx_ns = x_ns[1:] - x_ns[:-1]
    print(np.mean(dx_ns),np.median(dx_ns),scipy.stats.skew(dx_ns))
    plt.hist(dx_ns, bins = 30)
    plt.vlines(x=[np.mean(dx_ns)],ymin= 0, ymax= 50, linestyles='-',label='mean', colors=['pink'])
    plt.vlines(x=[np.median(dx_ns)],ymin= 0, ymax= 50, linestyles='-',label='median', colors=['orange'])
    plt.ylim([0, 45])
    plt.savefig('experiments/figure/dist/nvidia.png')

def lmarg_vs_sigmu(N=500,N0=0, save = True, step = 3, add = False, read = True, rep = 2, num_particles = 200,noise_sig=1, l=-0.05, k_v = 1e3, sigma_w = 2.8e-4, alpha=0.8, affix = '_3s', data='finance'):
    # n = 12
    # list_sigmu = np.linspace(-4, -1, num=n, endpoint=False)
    list_sigmu = np.array([-4.0,-3.0,-2.0,-1.0,0.0])
    # list_sigmu = np.array([-3.5])
    # list_sigmu = list_sigmu[6:]
    # list_sigmu = list_sigmu[:3]
    n = len(list_sigmu)
    if save:
        log_margs, mses, mse1s = np.zeros((n, rep)), np.zeros((n, rep)), np.zeros((n, rep))

        dic_lmarg = {}
        dic_mse = {}
        dic_mse1 = {}

        def is_inf_nan(nums):
            for num in nums:
                if num == np.inf or num == np.nan:
                    return True
            return False
        
        # inf_finance(num_particles=500, alpha=1.2, l=-0.05, k_v=1e3, sigma_mu=1e-3, sigma_w=2.8e-4)
        for i in range(n):
            print(list_sigmu[i])
            for j in range(rep):
                log_marg, mse, mse1,_ = inf_finance(step=step, N=N, data=data, noise_sig=noise_sig, num_particles=num_particles, alpha=alpha, l=l, k_v=k_v, sigma_mu=10**(list_sigmu[i]), sigma_w=sigma_w,N0=N0) # 1e-7
                if is_inf_nan([log_marg, mse, mse1]):
                    continue
                else:
                    log_margs[i,j], mses[i,j], mse1s[i,j] = log_marg, mse, mse1

            dic_lmarg[f'val{int(-list_sigmu[i]*10)}'] = log_margs[i,:]
            dic_mse[f'val{int(-list_sigmu[i]*10)}'] = mses[i,:]
            dic_mse1[f'val{int(-list_sigmu[i]*10)}'] = mse1s[i,:]

        # m_log_margs, var_log_margs = log_margs.mean(axis = 1), log_margs.var(axis = 1)
        # m_mses, var_mses = mses.mean(axis = 1), mses.var(axis = 1)

        # inf_finance_w_drift(num_particles=1000, alpha=0.8, l=-0.05, k_v=1e3, sigma_mu=1e-10, sigma_beta=1e-10, sigma_w=2.8e-4)
        log_marg_df, mse_df, mse1_df = pd.DataFrame(dic_lmarg), pd.DataFrame(dic_mse), pd.DataFrame(dic_mse1)
        log_marg_df.to_csv(f'experiments/data/lmarg/{data}/lmarg_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv', index=False)
        mse_df.to_csv(f'experiments/data/mse/{data}/mse_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv', index=False)
        mse1_df.to_csv(f'experiments/data/mse/{data}/mse_1s_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv', index=False)

    elif add:
        log_margs, mses, mse1s = np.zeros((n, rep)), np.zeros((n, rep)), np.zeros((n, rep))
        
        # inf_finance(num_particles=500, alpha=1.2, l=-0.05, k_v=1e3, sigma_mu=1e-3, sigma_w=2.8e-4)
        for i in range(n):
            for j in range(rep):
                log_margs[i,j], mses[i,j], mse1s[i,j],_ = inf_finance(noise_sig=noise_sig, step=step, data=data, N=N,N0=N0, num_particles=num_particles, alpha=alpha, l=l, k_v=k_v, sigma_mu=10**(list_sigmu[i]), sigma_w=sigma_w) # 1e-7

        dic_lmarg = pd.read_csv(f'experiments/data/lmarg/{data}/lmarg_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv').to_dict(orient='list')
        dic_mse = pd.read_csv(f'experiments/data/mse/{data}/mse_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv').to_dict(orient='list')
        dic_mse1 = pd.read_csv(f'experiments/data/mse/{data}/mse_1s_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv').to_dict(orient='list')

        for i in range(n):
            dic_lmarg[f'val{int(-list_sigmu[i]*10)}'] = np.append(dic_lmarg[f'val{int(-list_sigmu[i]*10)}'], log_margs[i,:]).flatten()
            dic_mse[f'val{int(-list_sigmu[i]*10)}'] = np.append(dic_mse[f'val{int(-list_sigmu[i]*10)}'], mses[i,:]).flatten()
            dic_mse1[f'val{int(-list_sigmu[i]*10)}'] = np.append(dic_mse1[f'val{int(-list_sigmu[i]*10)}'], mse1s[i,:]).flatten()

        log_marg_df, mse_df, mse1_df = pd.DataFrame(dic_lmarg), pd.DataFrame(dic_mse), pd.DataFrame(dic_mse1)

        log_marg_df.to_csv(f'experiments/data/lmarg/{data}/lmarg_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv', index=False)
        mse_df.to_csv(f'experiments/data/mse/{data}/mse_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv', index=False)
        mse1_df.to_csv(f'experiments/data/mse/{data}/mse_1s_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv', index=False)

    if read:
        log_margs = pd.read_csv(f'experiments/data/lmarg/{data}/lmarg_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv').to_numpy()
        mses = pd.read_csv(f'experiments/data/mse/{data}/mse_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv').to_numpy()
        mse1s = pd.read_csv(f'experiments/data/mse/{data}/mse_1s_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.csv').to_numpy()

        m_log_margs, var_log_margs = log_margs.mean(axis = 0), log_margs.var(axis = 0)
        m_mses, var_mses = mses.mean(axis = 0), mses.var(axis = 0)
        m_mse1s, var_mse1s = mse1s.mean(axis = 0), mse1s.var(axis = 0)
        print(m_log_margs, np.sqrt(var_log_margs))
        print(m_mses, np.sqrt(var_mses))
        print(m_mse1s, np.sqrt(var_mse1s))


        plt.figure()
        plt.xlabel(r'$\log_{10} \sigma_\mu$ values')
        plt.ylabel('log margs')
        plt.plot(list_sigmu, m_log_margs, label='log margs')
        plt.fill_between(list_sigmu, m_log_margs + np.sqrt(var_log_margs), m_log_margs - np.sqrt(var_log_margs), color='gray', alpha=0.2)
        plt.savefig(f'experiments/figure/simplified/marg/{data}/marg_vs_sigmu_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.png')

        plt.figure()
        plt.xlabel(r'$\log_{10} \sigma_\mu$ values')
        plt.ylabel(f'{step} step pred MSEs')
        plt.plot(list_sigmu, m_mses, label='MSEs')
        plt.fill_between(list_sigmu, m_mses + np.sqrt(var_mses), m_mses - np.sqrt(var_mses), color='gray', alpha=0.2)
        plt.savefig(f'experiments/figure/simplified/marg/{data}/mse_vs_sigmu_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.png')

        plt.figure()
        plt.xlabel(r'$\log_{10} \sigma_\mu$ values')
        plt.ylabel('1 step pred MSEs')
        plt.plot(list_sigmu, m_mse1s, label='MSEs')
        plt.fill_between(list_sigmu, m_mse1s + np.sqrt(var_mse1s), m_mse1s - np.sqrt(var_mse1s), color='gray', alpha=0.2)
        plt.savefig(f'experiments/figure/simplified/marg/{data}/mse_1s_vs_sigmu_n{num_particles}_kv{int(k_v)}_alpha{int(alpha*10)}{affix}.png')

    # fig, ax1 = plt.subplots()

    # # left y-axis: log_margs
    # color1 = 'tab:blue'
    # ax1.set_xlabel(r'$\log_{10} \sigma_\mu$ values')
    # ax1.set_ylabel('log margs', color=color1)
    # ax1.plot(list_sigmu, m_log_margs, color=color1, label='log margs')
    # ax1.fill_between(list_sigmu, m_log_margs + np.sqrt(var_log_margs), m_log_margs - np.sqrt(var_log_margs), color='gray', alpha=0.2)
    # ax1.tick_params(axis='y', labelcolor=color1)

    # # right y-axis: mses
    # ax2 = ax1.twinx()
    # color2 = 'tab:orange'
    # ax2.set_ylabel('MSEs', color=color2)
    # ax2.plot(list_sigmu, m_mses, color=color2, label='MSEs')
    # ax2.fill_between(list_sigmu, m_mses + np.sqrt(var_mses), m_mses - np.sqrt(var_mses), color='gray', alpha=0.2)
    # ax2.tick_params(axis='y', labelcolor=color2)

    # # optional: add a title and legend
    # fig.suptitle(r'Log-Marginal vs MSE across $\sigma_\mu$')
    # # if you want a combined legend, you can do:
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # plt.tight_layout()

def lmarg_vs_kvl(noise_sig = 1, N=500,N0=0, save = True, step = 3, sigma_mu = 1e-1, add = False, read = True, ite = 2, num_particles = 200, sigma_w = 2.8e-4, alpha=0.8, affix = '_3s', data='finance'):
    
    if data=='fish1d':
        ls = -np.array([1, 0.1, 0.01,0.001])
        # ls = -np.array([0.01])
        k_v = np.array([100,1000,2000,3000])
    else:
        ls = -np.array([10,1, 0.1, 0.01])
        k_v = np.array([0.1,1,10,100])
    X, Y = np.meshgrid(np.log10(k_v), np.log10(-ls), indexing='xy')

    log_margs, mses, mse3s = np.zeros([len(ls), len(k_v),ite]), np.zeros([len(ls), len(k_v),ite]), np.zeros([len(ls), len(k_v),ite])
    km, ksig, lm, lsig = 0.3, 1, 0.1, 1.5
    for k in range(ite):
        for i in range(len(ls)):
            for j in range(len(k_v)):
                log_margs[i,j,k], mse3s[i,j,k], mses[i,j,k],_ = inf_finance(sigma_mu=sigma_mu, num_particles=num_particles, sigma_w=sigma_w,  alpha=alpha, step=step, N=N, N0=N0,data=data, l = ls[i], k_v=k_v[j], noise_sig=noise_sig)
                log_margs[i,j,k] -= (np.log(k_v[j])-np.log(km))**2/(2*ksig**2) + np.log(k_v[j]) \
                        + (np.log(-1/ls[i])-np.log(1/lm))**2/(2*lsig**2) + np.log(1.5*np.sqrt(2*np.pi)) + 2*np.log(-ls[i])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    log_margs = log_margs.mean(axis = 2)
    surf = ax.plot_surface(X, Y, log_margs, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\log_{10}k_v$')
    ax.set_ylabel(r'$\log_{10} |\lambda|$')
    ax.set_zlabel(r'log marginal')
    maxid = np.argmax(log_margs)
    maxid = np.unravel_index(maxid, log_margs.shape)
    # import pdb;pdb.set_trace()
    print(f'max log marg: {log_margs[maxid]}, corresponding lambda: {ls[maxid[0]]}, k_v: {k_v[maxid[1]]}')


    ax = fig.add_subplot(2, 2, 2, projection='3d')
    mses = mses.mean(axis=2)
    surf2 = ax.plot_surface(X, Y, mses, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\log_{10}k_v$')
    ax.set_ylabel(r'$\log_{10}|\lambda|$')
    ax.set_zlabel(r'MSE')
    maxid = np.argmin(mses)
    maxid = np.unravel_index(maxid, mses.shape)
    print(f'min mse: {mses[maxid]}, corresponding lambda: {ls[maxid[0]]}, k_v: {k_v[maxid[1]]}')

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    mse3s = mse3s.mean(axis=2)
    surf2 = ax.plot_surface(X, Y, mse3s, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\log_{10}k_v$')
    ax.set_ylabel(r'$\log_{10}|\lambda|$')
    ax.set_zlabel(f'{step} step MSE')
    maxid = np.argmin(mse3s)
    maxid = np.unravel_index(maxid, mse3s.shape)
    print(f'min mse: {mse3s[maxid]}, corresponding lambda: {ls[maxid[0]]}, k_v: {k_v[maxid[1]]}')

    if save:
        plt.savefig(f'experiments/figure/simplified/{data}/lgmarg_mse.png')
    plt.show()

def gaussian_langevin_inf(N = 500, data = 'nvdia', l = -1, sigma_w = 0.1, k_v=1,return_logmarg = True, step = 5, N0=0, noise_sig=1):
    if data == 'finance':
        # c = 1e5
        datapath = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\dataEurUS.mat"
        x_ns, t_ns = extract_mat_data(datapath)
        start_time = 7.3259e5
        end_time = start_time + 0.1
        # import pdb;pdb.set_trace()
        mask = (t_ns>start_time) & (t_ns<end_time)
        x_ns = x_ns[mask]
        x_ns -= x_ns[0]
        t_ns =( t_ns[mask] - start_time)*1e3 #*1e4
        delta_ts = t_ns[1:] - t_ns[:-1]

    elif data == 'fish1d':
        datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'
        startx = 3000+N0
        tracks = extract_track(datapath)
        dims=0
        x_ns = tracks[0,startx:(startx+N),dims].T
        x_ns -= x_ns[0]
        delta_ts = 1
        t_ns = np.linspace(0, len(x_ns),endpoint=False,num=len(x_ns))

    elif data == 'nvdia_tl':
        csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVDA.USUSD_Ticks_20.06.2025-20.06.2025.csv"
        nvda_data_reloaded= pd.read_csv(csvfile)
        x_ns = (np.array(nvda_data_reloaded['Ask'])+np.array(nvda_data_reloaded['Bid']))/2

        nvda_data_reloaded['Local time'] = pd.to_datetime(nvda_data_reloaded['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
        nvda_data_reloaded['Local time'] = (nvda_data_reloaded['Local time'] - nvda_data_reloaded['Local time'].iloc[0]).dt.total_seconds()

        t_ns = np.array(nvda_data_reloaded['Local time'])

        x_ns, t_ns = x_ns[N0:N0+N]-x_ns[N0], t_ns[N0:N0+N]
        delta_ts = t_ns[1:] - t_ns[:-1]

    else:
        csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"
        nvda_data_reloaded= pd.read_csv(csvfile, index_col=0, parse_dates=True)
        x_ns = np.array(nvda_data_reloaded['Price']['2025-01-29'])
        x_ns = (x_ns - x_ns[0])
        x_ns = np.flip(x_ns) # pick the prev 300 data for const mu
        N=len(x_ns)
        delta_ts = 1
        t_ns = np.linspace(0, len(x_ns),endpoint=False,num=len(x_ns))

    T = 1

    if data == 'fish1d':
        rng = default_rng(42)
        b = rng.normal(0.0, noise_sig, N)
        y_ns = x_ns  + b
    else:
        y_ns = x_ns # +np.random.normal(0, sigma_w*k_v, N)

    n_mus, n_vars, n_log_ws, E_ns, log_marg, mse, mse3, y_1step, hr, hr3 = gaussian_pf_1d(y_ns,x_ns, sigma_w, k_v*sigma_w, l, delta_ts, step=step)
    # print(E_ns.mean())
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w, sigmaw_range=[0.8, 1.3])
    # n_mus, n_vars, n_log_ws, E_ns, log_marg = particle_filter_1d_w_drift(y_ns, num_particles, c, T, sigma_mu, sigma_beta,sigma_w, k_v*sigma_w, alpha, l, delta_ts)
    if return_logmarg:
        return log_marg, mse/(len(y_ns)-1), mse3/(len(y_ns)-step), np.mean((average[:,0]-x_ns)**2)
    else:
        print(f'log marg: {log_marg},  mse3: {mse3/(len(y_ns)-step)}, mse: {mse/(len(y_ns)-1)}, mse0:{np.mean((average[:,0]-x_ns)**2)}')
        print(f'hit rate3: {hr3/(len(y_ns)-step)}, hr: {hr/(len(y_ns)-1)}')

    
    # with open(f'experiments/figure/wdrift/{data}/marginals.txt', 'a') as f:
    #     line = f'l={l}, c={c}, N={N}, dt={delta_ts}, sigma w ={sigma_w}, sigma mu={sigma_mu}, alpha = {alpha}, kv={k_v} \nlog marginals:{log_marg}\n'
    #     f.write(line)
    # np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\finance",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, marg = marg, allow_pickle=True)

    plt.figure(figsize=(8,5))
    if data == 'nvdia_tl':
        plt.ylabel('Stock Price')
        plt.xlabel('Seconds')
    elif data == 'fish1d':
        plt.ylabel('Displacement')
        plt.xlabel('n')
    pred_xs = average[:,:2]
    # plt.plot(t_ns, pred_xs[:,0])
    plt.plot(t_ns, y_1step,label='1 step pred')
    if data == 'fish1d':
        plt.plot(t_ns, x_ns, label = 'True')
        plt.scatter(t_ns, y_ns, color='pink',s=5, label='Noisy Obs')
    else:
        plt.scatter(t_ns, x_ns,color='red',s=5, label='Noisy Obs')

    if data == 'nvdia_tl':
        miny = -0.3
        plt.ylim([miny,0.09])
        plt.fill_between(t_ns, miny*np.ones_like(t_ns), miny+np.abs(y_1step- x_ns),label='Error',
                 color='gray', alpha=0.2)
    # plt.plot(t_ns, y_nstep- x_ns, linestyle = '--', label = '3 step pred')
    # plt.ylim([min(y_1step)-1,max(y_1step)+0.5])
    # plt.ylim([-4.5,4])
    # plt.fill_between(t_ns, (-4.5)*np.ones_like(t_ns), -4.5+np.abs(y_1step- x_ns),label='Error',
    #              color='gray', alpha=0.2)
    plt.legend()
    plt.savefig(f'experiments/figure/gaussian/{data}/xs_kv{int(k_v)}_l{int(l)}_1err.png')
    plt.show()

    # return
    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    if data == 'nvdia_tl':
        plt.ylabel('Stock Price')
        plt.xlabel('Seconds')
    elif data == 'fish1d':
        plt.ylabel('Displacement')
        plt.xlabel('n')
    pred_xs = average[:,:2]
    plt.plot(t_ns, pred_xs[:,0],label='Particle Mean')
    # plt.plot(t_ns, x_ns, linestyle = '--', color = 'red')
    if data == 'fish1d':
        plt.plot(t_ns, x_ns, label = 'True')
        plt.scatter(t_ns, y_ns, color='pink',s=5, label='Noisy Obs')
    else:
        plt.scatter(t_ns, x_ns,color='pink',s=5, label='Noisy Obs')
    # plt.plot(t_ns, y_1step, linestyle = '--', color = 'red')
    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(t_ns, average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    # plt.legend(['Particle mean','Data'])
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.ylabel('velocity')
    plt.plot(t_ns, pred_xs[:,1])
    plt.hlines([0.0], t_ns[0], t_ns[-1],linestyle = '--', color = 'green')
    if data == 'nvdia_tl':
        plt.ylim([-0.1,0.1])
    else:
        plt.ylim([-0.3,0.3])
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # plt.ylim([min(average[25:,1] - std3[25:,1]),max(average[25:,1] + std3[25:,1])])
    
    # plt.fill_between(t_ns, average[:,1] - std3[:,1], average[:,1] + std3[:,1],
    #              color='gray', alpha=0.2)
    
    plt.savefig(f'experiments/figure/gaussian/{data}/xv_kv{int(k_v)}_l{int(l)}.png')
    plt.show()

    return
    plt.figure(figsize=(8,5))
    plt.ylabel('Centered Stock Price')
    plt.xlabel('Minutes')
    pred_xs = average[:,:2]
    plt.plot(t_ns, pred_xs[:,0])
    # plt.plot(t_ns, pred_y, linestyle = '--', color = 'red')
    plt.scatter(t_ns, x_ns,color='pink',s=5)
    plt.ylim([min(min(average[25:,0] - std3[25:,0]), min(x_ns))-0.5,max(max(average[25:,0] + std3[25:,0]), max(x_ns))+0.5])
    plt.fill_between(t_ns, average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    plt.legend(['Particle mean','Data'])
    
    # plt.subplot(3,1,2)
    # plt.ylabel('velocity')
    # plt.plot(t_ns, pred_xs[:,1])
    # # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # # plt.ylim([min(average[25:,1] - std3[25:,1]),max(average[25:,1] + std3[25:,1])])
    # plt.ylim([-0.5,0.5])
    # plt.fill_between(t_ns, average[:,1] - std3[:,1], average[:,1] + std3[:,1],
                #  color='gray', alpha=0.2)
    
    plt.savefig(f'experiments/figure/gaussian/{data}/xs_kv{int(k_v)}_l{int(l)}.png')
    print('saved')

    plt.figure()
    plt.plot(xs, fxs)
    # plt.axvline(x = sigma_w**2, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2$')
    plt.ylabel('Posterior')
    plt.savefig(f'experiments/figure/gaussian/{data}/sigma_kv{int(k_v)}_l{int(l)}.png')

def gaussian_lgmarg_mse(step = 3, N = 1000, N0=0, data = 'nvdia',ite = 3,save=True, dim=2, sigma_w = 1, noise_sig=1):
    if dim==2:
        km, ksig, lm, lsig = 0.3, 1, 0.1, 1.5

        if data=='fish1d':
            ls = -np.array([1, 0.1, 0.01,0.001])
            k_v = np.array([10,100, 1000])
        else:
            # ls = -np.array([10,1, 0.1, 0.01])
            # k_v = np.array([0.01,0.1,1,10])

            ls = -np.array([2,1.5,1.0,0.5])
            k_v = np.array([0.05,0.1,0.2,0.4])

        X, Y = np.meshgrid(np.log10(k_v), np.log10(-ls), indexing='xy')

        log_margs, mses, mse3s = np.zeros([len(ls), len(k_v),ite]), np.zeros([len(ls), len(k_v),ite]), np.zeros([len(ls), len(k_v),ite])
        
        for k in range(ite):
            for i in range(len(ls)):
                for j in range(len(k_v)):
                    log_margs[i,j,k], mses[i,j,k], mse3s[i,j,k], _ = gaussian_langevin_inf(step=step, sigma_w=sigma_w, N=N,N0=N0, data=data, l = ls[i], k_v=k_v[j], noise_sig=noise_sig)
                    # import pdb;pdb.set_trace()
                    log_margs[i,j,k] -= (np.log(k_v[j])-np.log(km))**2/(2*ksig**2) + np.log(k_v[j]) \
                        + (np.log(-1/ls[i])-np.log(1/lm))**2/(2*lsig**2) + np.log(1.5*np.sqrt(2*np.pi)) + 2*np.log(-ls[i])

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        log_margs = log_margs.mean(axis = 2)
        surf = ax.plot_surface(X, Y, log_margs, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\log_{10}k_v$')
        ax.set_ylabel(r'$\log_{10} |\lambda|$')
        ax.set_zlabel(r'log marginal')
        maxid = np.argmax(log_margs)
        maxid = np.unravel_index(maxid, log_margs.shape)
        # import pdb;pdb.set_trace()
        print(f'max log marg: {log_margs[maxid]}, corresponding lambda: {ls[maxid[0]]}, k_v: {k_v[maxid[1]]}')


        ax = fig.add_subplot(2, 2, 2, projection='3d')
        mses = mses.mean(axis=2)
        surf2 = ax.plot_surface(X, Y, mses, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\log_{10}k_v$')
        ax.set_ylabel(r'$\log_{10}|\lambda|$')
        ax.set_zlabel(r'MSE')
        maxid = np.argmin(mses)
        maxid = np.unravel_index(maxid, mses.shape)
        print(f'min mse: {mses[maxid]}, corresponding lambda: {ls[maxid[0]]}, k_v: {k_v[maxid[1]]}')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        mse3s = mse3s.mean(axis=2)
        surf2 = ax.plot_surface(X, Y, mse3s, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\log_{10}k_v$')
        ax.set_ylabel(r'$\log_{10}|\lambda|$')
        ax.set_zlabel(f'{step} step MSE')
        maxid = np.argmin(mse3s)
        maxid = np.unravel_index(maxid, mse3s.shape)
        print(f'min mse: {mse3s[maxid]}, corresponding lambda: {ls[maxid[0]]}, k_v: {k_v[maxid[1]]}')

        if save:
            plt.savefig(f'experiments/figure/gaussian/{data}/lgmarg_mse.png')
        plt.show()
    if dim == 1:
        ls = -np.array([5,4,3,2, 1, 0.75,0.5,0.25, 0.1, 0.01,0.001])
        k_v = 0.5

        log_margs, mses, mse3s = np.zeros([len(ls),ite]), np.zeros([len(ls),ite]) , np.zeros([len(ls),ite])
        
        for k in range(ite):
            for i in range(len(ls)):
                log_margs[i,k], mses[i,k], mse3s[i,k] = gaussian_langevin_inf(step=step, N=N,N0=N0, data=data, l = ls[i], k_v=k_v)

        fig, ax1 = plt.subplots()

        # left y-axis: log_margs
        color1 = 'tab:blue'
        ax1.set_xlabel(r'$\log_{10} |\lambda|$ values')
        ax1.set_ylabel('log margs', color=color1)
        ax1.plot(np.log10(-ls), log_margs.mean(axis=-1), color=color1, label='log margs')
        # ax1.fill_between(list_sigmu, m_log_margs + np.sqrt(var_log_margs), m_log_margs - np.sqrt(var_log_margs), color='gray', alpha=0.2)
        ax1.tick_params(axis='y', labelcolor=color1)

        # right y-axis: mses
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('MSEs', color=color2)
        ax2.plot(np.log10(-ls), mses.mean(axis=-1), color=color2, label='MSEs')
        ax2.plot(np.log10(-ls), mse3s.mean(axis=-1), color='tab:pink', label=f'{step} step MSEs')
        # ax2.fill_between(list_sigmu, m_mses + np.sqrt(var_mses), m_mses - np.sqrt(var_mses), color='gray', alpha=0.2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # optional: add a title and legend
        fig.suptitle(r'Log-Marginal vs MSE across $\lambda$')
        # if you want a combined legend, you can do:
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.tight_layout()
        if save:
            plt.savefig(f'experiments/figure/gaussian/{data}/lgmarg_mse_lonly.png')
        plt.show()

def multi_inf(k=10):
    data = np.zeros((4,k))
    for i in range(k):
        # data[0,i],data[1,i],data[2,i], data[3,i] =  inf_finance(data='fish1d',num_particles=200, N0=500, N =1000, alpha=0.8, l=-0.01, k_v=1000, sigma_mu=10**(-3.0), sigma_w=1,c=10, returnlmarg=True, step=3,noise_sig=1)
        data[0,i],data[1,i],data[2,i], data[3,i] = inf_finance(data='nvdia_tl',num_particles=1000, alpha=0.8, l=-1, k_v=1, sigma_mu=10**(-2), sigma_w=2.5,c=10, returnlmarg=True, step=3, N=500,N0 =20000)
        # data[0,i],data[1,i],data[2,i]= gaussian_langevin_inf(k_v = 10, l=-0.001, return_logmarg=True, data='fish1d', step=10, N0=500, N=1000, sigma_w=1, noise_sig=1)
    print(data.mean(axis=1), data.std(axis=1))
    print(data[0,:].max(), data[1,:].min(), data[2,:].min(), data[3,:].min(), )

if __name__=='__main__':
    # marg_wrt_l(alpha=1.2, num_particles=100, k=5, save=True)
    # alpha = 1.2

    # multi_inf()
    # x = simu_2d(save= False, alpha= alpha, sigma_w=0.05, k_v=1e3) # 1e4 for 0.9
    # x = simu_2d_w_drift(save= True, alpha= alpha, sigma_w=0.05, k_v=5e2)
    # # # plot histogram of velocity increments
    # # vs= x[1:,1] - x[:-1, 1]
    # # plt.figure()
    # # plt.hist(vs, bins = 50)
    # # plt.show()

    # # test_data(num_particles=100)
    # test_data_2d_wdrift(num_particles=1000, alpha=alpha)
    # plot_result_from_stored_wdrift(alpha=alpha)
    
    # lmarg_vs_kvl(N=500, N0=20500,num_particles=100, ite=3, step=3, sigma_mu=10**(-2), sigma_w=1, alpha=0.8, data='nvdia_tl', affix='')
    lmarg_vs_sigmu(num_particles=100, rep= 3, step=3,N=500,N0=20500,  data='nvdia_tl', save=True, add=True, k_v=0.1, sigma_w=1.0,l=-10, alpha=0.8, affix="_l1") #500 k_v=0.1, sigma_w=1,l=-1
    # inf_finance(data='finance',num_particles=500, alpha=0.8, l=-0.005, k_v=100, sigma_mu=10**(-2), sigma_w=1,c=20, returnlmarg=False, step=10)
    # inf_finance(data='nvdia',num_particles=200, alpha=0.8, l=-0.001, k_v=100, sigma_mu=10**(-3.5), sigma_w=2.5,c=10, returnlmarg=False, step=10)
    # inf_finance(data='nvdia_tl',num_particles=500, alpha=0.8, l=-1, k_v=1, sigma_mu=10**(-2), sigma_w=2.5,c=10, returnlmarg=False, step=3, N=500,N0 =20000, K=10)# l=-1, k_v=1 log marg: 1560.0504093668767, mse: 0.0035070204068031207, mse1: 0.0006875091312458532


    # lmarg_vs_kvl(ite=2,num_particles=20, N =500, alpha=0.8, sigma_mu=10**(-2.0), sigma_w=1,  step=10,noise_sig=1, data='fish1d')
    # lmarg_vs_sigmu(rep=3,data='fish1d',num_particles=200, N =500, alpha=0.8, l=-0.01, k_v=1000, sigma_w=1, step=10,noise_sig=1, affix="_l001") #500 k_v=100, sigma_w=0.1,l=-0.005
    # inf_finance(data='fish1d',num_particles=500, N0=500, N =1000, alpha=0.8, l=-0.01, k_v=1000, sigma_mu=10**(-2.0), sigma_w=1,c=10, returnlmarg=False, step=3,noise_sig=1) # log marg: 2687.212888159273, mse: 0.5989060250308442, mse1: 0.0038112799159313255
    # l=-1, kv=1, sigmaw=1e-1,c=10*

    # gaussian_lgmarg_mse(ite=1,save=True,dim=2, data='fish1d', step=10, N = 500,N0=0, sigma_w=1)
    # gaussian_langevin_inf(k_v = 10, l=-0.1, return_logmarg=False, data='fish1d', step=3, N0=500, N=1000, sigma_w=1, noise_sig=1) # seems only fair to fix a k_v value (noise variance)

    # gaussian_lgmarg_mse(ite=1,save=True,dim=2, data='nvdia_tl', step=3, N = 500,N0=20500)
    # gaussian_langevin_inf(k_v = 10, l=-0.1, return_logmarg=False, data='nvdia', step=10, N=-1)
    # gaussian_langevin_inf(k_v = 0.1, l=-1, return_logmarg=False, data='nvdia_tl', step=3, N=500, N0=20000, sigma_w=1) #k_v = 0.1, l=-1，N=500 log marg: 1580.0696213389592,  mse3: 0.0014704484428881956, mse: 0.0005610093089112554
    # gaussian_langevin_inf(k_v = 0.1, l=-1, return_logmarg=False, data='finance', step=10, N=-1) # log marg: 7466.522673705947,  mse3: 4.3980425168024695e-08, mse: 4.540358507578895e-09
    # lambda: -1, k_v: 0.1, max log marg: 7466.5226737059465, min mse: 4.534874982811287e-09, min 3 mse: 1.1898598400377809e-08



    # inf_1d_fish(num_particles = 200, N=1200,k_v=1000, sigma_mu=1e-2,l=-1e-4, alpha=0.8, noise_sig=1)
    # inf_2d_fish(num_particles=200, N=1200, m=400, k_v=1000, sigma_mu=1e-6,l=-0.01, alpha=0.8)
    # inf_2d_fish_wdrift(num_particles=500 , N=1200, m=400)
    # inf_3d_fish(alpha= 0.9, num_particles =200, N=1200,m=200,scale=800)

    # dist_nvidia()
