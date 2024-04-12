import numpy as np
import matplotlib.pyplot as plt
from tools import *
import scipy

def resample(log_weight_p, mu_p, var_p, E_ns):
    """resample particles, input/output arrays"""
    P = len(log_weight_p)
    indices = np.random.choice(list(range(P)), size = P, p = np.exp(log_weight_p))
    log_weight_p = np.log(np.ones(P)*(1/P))
    return mu_p[indices,:], var_p[indices,:,:], E_ns[indices,:], log_weight_p

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
    K = var_n_prev_n @ obs_matrix.T @ np.linalg.inv(sigma_n_prev_n)
    mu_n_n = mu_n_prev_n + K@ (y_n - y_hat_n_prev_n)
    var_n_n = (np.identity(len(mu))-K@C)@var_n_prev_n
    
    return mu_n_n, var_n_n, sigma_n_prev_n, y_hat_n_prev_n
    

def particle_filter_1d(y_ns, P, c, T, sigma_mu, sigma_w, noise_sig, alpha,alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    n_mus = np.zeros((N, P, 5))
    n_vars = np.zeros((N, P, 5, 5))
    n_log_ws = np.zeros((N, P))

    # initialize x
    n_vars[0] = np.array([np.matrix(np.identity(5))*10]*P) # size: (P, 5, 5)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((P, N+1)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.array([1,0,-alpha/(alpha-1) * c**(1-1/alpha),0,0])

    for n in range(N):
        n = n+1
        y_n = y_ns[n]
        
        if n%5 == 0: # resample
            n_log_ws[n-1], n_mus[n-1], n_vars[n-1], n_log_ws[n-1] = resample(n_log_ws[n-1], n_mus[n-1], n_vars[n-1], n_log_ws[n-1])
        
        # update
        for p in range(P):
            gammas, vs = generate_jumps(c, T)
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
            
            # transition matrices
            C = noise_variance_C(processer.S_mu(sigma_mu),processer.S_s_t(sigma_w), delta_t, l, sigma_w)
            A = transition_matrix(processer.mean_s_t(), delta_t, l)
            
            # kalman filter update
            n_mus[n,p,:], n_vars[n,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter(A, C, observation_matrix, noise_sig, n_mus[n-1,p,:], n_vars[n-1,p,:,:], y_n)
            
            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[p,n] = -(y_n-y_hat_n_prev_n).T*np.linalg.inv(norm_sigma_n_prev_n)*(y_n-y_hat_n_prev_n)/2
            beta_w_post_p = beta_w - sum(E_ns[p,:])
            log_like = -0.5*np.log(np.linalg.det(sigma_n_prev_n))-(alpha_w+n*2/2)*np.log(beta_w_post_p)\
                    +((n-1)*2/2+alpha_w)*np.log(beta_w - sum(E_ns[p,:n]))+\
                    scipy.special.loggamma(n*2/2+alpha_w)-scipy.special.loggamma((n-1)*2/2+alpha_w) # -2/2*np.log(2*np.pi)
            n_log_ws[n,p] = n_log_ws[n-1, p]+ log_like

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns

def process_filter_results(n_mus, n_vars, n_log_ws, E_ns, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    # examine data
    (N, num_particles, D) = n_mus.shape()
    
    average = np.zeros((N, D))
    std3 = np.zeros((N, D))
    avg_P = np.zeros((N, D, D))

    mean_sigmas = np.zeros((N,num_particles))
    var_sigmas = np.zeros((N,num_particles))
    betas = np.zeros((N, num_particles))
    for n in range(N):
        alpha_n = alpha_w + (n+1)
        for p in range(num_particles):
            betas[n,p] = beta_w - sum(E_ns[p,:(n+2)])
            mean_sigmas[n,p] = betas[n,p]/(alpha_n-1)
            var_sigmas[n,p] = max(0, betas[n,p]**2/(alpha_n-1)**2/(alpha_n-2))

    tot_mean_sigma = np.zeros(N)
    tot_var_sigma = np.zeros(N)

    for i in range(N):
        alpha_n = alpha_w + (i+1)
        for d in range(D):
            average[i,d]=np.dot(np.array(n_mus[i,:,d,:].T)[0], np.exp(n_log_ws[i,:]))
            for j in range(D):
                if i<=2:
                    var_x_ij = n_vars[i,:,d,j]*sigma_w**2
                else:
                    var_x_ij = n_vars[i,:,d,j]*betas[i,:]/(alpha_n-1)
                avg_P[i,d,j] = np.dot(np.array(var_x_ij.T), np.exp(n_log_ws[i,:]))+\
                np.dot((np.array(n_mus[i,:,d,:].T)[0]-average[i,d])*(np.array(n_mus[i,:,j,:].T)[0]-average[i,j]), np.exp(n_log_ws[i,:]))
                
            std3[i,d]=np.sqrt(avg_P[i,d,d])*3

        tot_mean_sigma[i]=np.dot(mean_sigmas[i,:], np.exp(n_log_ws[i,:]))
        tot_var_sigma[i] = np.dot(var_sigmas[i,:], np.exp(n_log_ws[i,:])) + np.dot((mean_sigmas[i,:]-tot_mean_sigma[i])*(mean_sigmas[i,:]-tot_mean_sigma[i]),np.exp(n_log_ws[i,:]))

    return average, std3, betas, tot_mean_sigma, tot_var_sigma

    
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
    
    data_read = np.load('experiments/data/x_ns.npz')
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    y_ns = data_read['y'][:,0] # observation, size (N, 2)
    
    num_particles = 100
    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, noise_sig = k_v*sigma_w)
    average, std3, betas,_,_ = process_filter_results(n_mus, n_vars, n_log_ws, E_ns)