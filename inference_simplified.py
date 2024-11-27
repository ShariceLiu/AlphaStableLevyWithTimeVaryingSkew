import numpy as np
import matplotlib.pyplot as plt
from tools.tools_sim import *
import scipy
# from simulation import forward_simulation_1d_w_integrals
from scipy.stats import invgamma
from tqdm import tqdm
from real_data import extract_track, extract_mat_data

def resample(log_weight_p, mu_p, var_p, E_ns):
    """resample particles, input/output arrays"""
    P = len(log_weight_p)
    indices = np.random.choice(list(range(P)), size = P, p = np.exp(log_weight_p))
    log_weight_p = np.log(np.ones(P)*(1/P))
    return log_weight_p, mu_p[indices,:], var_p[indices,:,:], E_ns[indices]

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
    # import pdb;pdb.set_trace()
    
    return mu_n_n, var_n_n, sigma_n_prev_n, y_hat_n_prev_n

def particle_filter_1d(y_ns, P, c, T, sigma_mu, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """
    P: number of particles
    T: 
    """
    N = len(y_ns)
    n_mus = np.zeros((N, P, 3))
    n_vars = np.zeros((N, P, 3, 3))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    var_0 = np.identity(3)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 3, 3)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.array([1,0,0])

    # time
    if isinstance(delta_t, float) or isinstance(delta_t, int):
        delta_ts = np.ones([N-1])*delta_t
    else: # iterative object
        delta_ts = delta_t

    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
        
        delta_t_n = delta_ts[n]
        # update
        for p in range(P):
            vs, gammas = generate_jumps(c, T, delta_t_n)
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t_n, c, T, l)
            
            # transition matrices
            C = noise_variance_C(processer, sigma_w, sigma_mu)
            A = transition_matrix(processer)
            
            # kalman filter update
            n_mus[m,p,:], n_vars[m,p,:,:], sigma_n_prev_n, y_hat_n_prev_n = kalman_filter(A, C, observation_matrix, noise_sig, n_mus[n,p,:], n_vars[n,p,:,:], y_n)
            
            # update log weight
            norm_sigma_n_prev_n = sigma_n_prev_n /sigma_w**2
            E_ns[m,p] = -(y_n-y_hat_n_prev_n)**2/(norm_sigma_n_prev_n)/2
            beta_w_post_p = beta_w - sum(E_ns[:,p])
            log_like = -0.5*np.log(sigma_n_prev_n)-(alpha_w+m/2)*np.log(beta_w_post_p)\
                    +((m-1)/2+alpha_w)*np.log(beta_w - sum(E_ns[:m,p]))+\
                    scipy.special.loggamma(m/2+alpha_w)-scipy.special.loggamma(n/2+alpha_w) # -2/2*np.log(2*np.pi)
            n_log_ws[m,p] = n_log_ws[n, p]+ log_like

        # normalise weights
        n_log_ws[m,:] = n_log_ws[m,:]- np.log(sum(np.exp(n_log_ws[m,:])))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns

def process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w, alpha_w = 0.000000000001,beta_w = 0.000000000001):
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
            betas[n,p] = beta_w - sum(E_ns[:(n+2),p])
            mean_sigmas[n,p] = betas[n,p]/(alpha_n-1)
            var_sigmas[n,p] = max(0, betas[n,p]**2/(alpha_n-1)**2/(alpha_n-2))

    tot_mean_sigma = np.zeros(N)
    tot_var_sigma = np.zeros(N)
    
    if D == 5: # test if it's 2 dim
        d_a = 1
    else:
        d_a = 2

    for i in range(N):
        alpha_n = alpha_w + i/d_a
        for d in range(D):
            average[i,d]=np.dot(n_mus[i,:,d], np.exp(n_log_ws[i,:]))
            for j in range(D):
                if i<=2:
                    var_x_ij = n_vars[i,:,d,j]*sigma_w**2
                else:
                    var_x_ij = n_vars[i,:,d,j]*betas[i,:]/(alpha_n-1/d_a)
                avg_P[i,d,j] = np.dot(var_x_ij, np.exp(n_log_ws[i,:]))+\
                np.dot((n_mus[i,:,d]-average[i,d])*(n_mus[i,:,j]-average[i,j]), np.exp(n_log_ws[i,:]))
                
            std3[i,d]=np.sqrt(avg_P[i,d,d])*3

        tot_mean_sigma[i]=np.dot(mean_sigmas[i,:], np.exp(n_log_ws[i,:]))
        tot_var_sigma[i] = np.dot(var_sigmas[i,:], np.exp(n_log_ws[i,:])) + np.dot((mean_sigmas[i,:]-tot_mean_sigma[i])*(mean_sigmas[i,:]-tot_mean_sigma[i]),np.exp(n_log_ws[i,:]))
    
    alpha = alpha_w + N/d_a
    xs = np.linspace(sigma_w**2*0.8,sigma_w**2*1.3,100)
    fxs = 0
    for p in range(num_particles):
        x_beta = xs/betas[-1,p]
        pdf = invgamma.pdf(x_beta, alpha)/betas[-1,p]*np.exp(n_log_ws[-1,p])
        fxs += pdf

    return average, std3, betas, xs, fxs

def inf_1d_fish(num_particles = 100, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -1e-2
    c = 10
    N = 100
    startx = 3000
    delta_t = 2
    sigma_w = 0.1
    sigma_mu = 1e-3
    alpha = 1.6
    k_v = 10 # 1.5e4 for alpha=0.9

    x_ns = extract_track(datapath)[0,startx:(startx+N),0]
    # add noise
    y_ns = x_ns + +np.random.normal(0, sigma_w*k_v, N)
    num_particles = 200

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish1d",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('displacement')
    pred_xs = average[:,:2]
    plt.plot(pred_xs[:,0])
    plt.plot(x_ns, linestyle = '--', color = 'red')

    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(range(len(average)), average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    
    plt.scatter(range(N),y_ns,color='orange',s=5)
    plt.legend(['pred','true','noisy'])

    plt.subplot(2,1,2)
    plt.ylabel('mu')
    plt.plot(average[:,-1])
    plt.ylim([min(average[25:,-1] - std3[25:,-1]),max(average[25:,-1] + std3[25:,-1])])
    plt.fill_between(range(len(average)), average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/fish/xs_{int(alpha*10)}_l{int(abs(l))}.png')

def inf_finance(num_particles = 200):
    datapath = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\dataEurUS.mat"

    l = -1e-2
    c = 10
    N = 500
    T = 1
    sigma_w = 0.05
    sigma_mu = 2e-3
    alpha = 1.9
    k_v = 200

    x_ns, t_ns = extract_mat_data(datapath)
    x_ns = x_ns[:N]*1e4 - x_ns[0]*1e4
    t_ns = t_ns[:N]*1e4
    delta_ts = t_ns[1:] - t_ns[:-1]

    # add noise
    y_ns = x_ns # +np.random.normal(0, sigma_w*k_v, N)
    # plt.plot(t_ns, x_ns)
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # plt.savefig(r'experiments\figure\real_data\finance\noisy')


    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, T, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_ts)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    # np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\finance",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('displacement')
    pred_xs = average[:,:2]
    plt.plot(t_ns, pred_xs[:,0])
    # plt.plot(t_ns, x_ns, linestyle = '--', color = 'red')
    plt.scatter(t_ns, y_ns,color='orange',s=5)
    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(t_ns, average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    
    plt.legend(['pred','noisy'])
    plt.subplot(2,1,2)
    plt.ylabel('mu')
    plt.plot(t_ns, average[:,-1])
    # plt.ylim([min(average[25:,-1] - std3[25:,-1]),max(average[25:,-1] + std3[25:,-1])])
    plt.ylim([-np.mean(np.abs(average[25:,-1]))*3,np.mean(np.abs(average[25:,-1]))*3])
    plt.fill_between(t_ns, average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/finance/xs_{int(alpha*10)}_l{int(abs(l))}.png')
    
if __name__=='__main__':
    # inference_filtering2d(num_particles = 100, datapath = 'experiments/data/2d/x_ns9.npz') 
    # plot_result_from_stored()

    inf_finance()
    # inf_1d_fish()
