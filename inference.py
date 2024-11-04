import numpy as np
import matplotlib.pyplot as plt
from tools import *
import scipy
# from simulation import forward_simulation_1d_w_integrals
from scipy.stats import invgamma
from tqdm import tqdm
from real_data import extract_track

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

def kalman_filter2d(A, C, obs_matrix, noise_sig, mu, var, y_n):
    """
    consider state transition and observation like: 
    x_t = A x_s + e_1, e_1 ~ N(0, C)
    y_t = obs_matrix x_t + e_2, e_2 ~ N(0, noise_sig^2*I)
    """
    mu_n_prev_n = A@mu
    var_n_prev_n = A@var@A.T + C
    y_hat_n_prev_n = obs_matrix@mu_n_prev_n
    sigma_n_prev_n = obs_matrix@var_n_prev_n @ obs_matrix.T + noise_sig**2 * np.identity(2)
    if (np.linalg.inv(sigma_n_prev_n) == np.inf).any():
        import pdb;pdb.set_trace()
    K = var_n_prev_n @ obs_matrix.T @ np.linalg.inv(sigma_n_prev_n)
    mu_n_n = mu_n_prev_n + K@ (y_n - y_hat_n_prev_n)
    var_n_n = (np.identity(len(mu))-K@obs_matrix)@var_n_prev_n
    # import pdb;pdb.set_trace()
    
    return mu_n_n, var_n_n, sigma_n_prev_n, y_hat_n_prev_n
    

def particle_filter_1d(y_ns, P, c, T, sigma_mu, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    n_mus = np.zeros((N, P, 5))
    n_vars = np.zeros((N, P, 5, 5))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    var_0 = np.identity(5)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 5, 5)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.array([1,0,-alpha/(alpha-1) * c**(1-1/alpha),0,0])

    for n in tqdm(range(N-1)):
        m = n+1
        y_n = y_ns[m]
        
        if n%2 == 0: # resample
            n_log_ws[n], n_mus[n], n_vars[n], E_ns[n] = resample(n_log_ws[n], n_mus[n], n_vars[n], E_ns[n])
        
        # update
        for p in range(P):
            vs, gammas = generate_jumps(c, T)
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
            
            # transition matrices
            C = noise_variance_C(processer.S_mu(sigma_mu),processer.S_s_t(sigma_w), processer.hi_fi_intQ(sigma_mu,l), processer.hi_fi_vi(sigma_mu),delta_t, l, sigma_mu)
            A = transition_matrix(processer.mean_s_t(), delta_t, l)

            # C = noise_Cs[m]
            # A = trans_As[m]
            
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

def particle_filter_2d(y_ns, P, c, T, sigma_mus, sigma_w, noise_sig, alpha,l, delta_t, trans_As = None, noise_Cs=None, alpha_w = 0.000000000001,beta_w = 0.000000000001):
    """P: number of particles"""
    N = len(y_ns)
    n_mus = np.zeros((N, P, 10))
    n_vars = np.zeros((N, P, 10, 10))
    n_log_ws = np.zeros((N, P))
    n_log_ws[0,:] = np.log(np.ones(P)*(1/P))

    # initialize x
    var_0 = np.identity(10)*5
    n_vars[0] = np.array([var_0]*P) # size: (P, 10, 10)
    n_vars[1] = np.array([var_0]*P)
    n_log_ws[0] = np.log(np.ones(P)*(1/P))
    E_ns = np.zeros((N+1,P)) # store exp likelihood of y
    
    # some useful constants
    observation_matrix = np.zeros((2,10))
    observation_matrix[0, :5] = np.array([1,0,-alpha/(alpha-1) * c**(1-1/alpha),0,0])
    observation_matrix[1, 5:] = np.array([1,0,-alpha/(alpha-1) * c**(1-1/alpha),0,0])

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
            vs, gammas = generate_jumps(c, T)
            processer = alphaStableJumpsProcesser(gammas, vs, alpha, delta_t, l)
            
            # transition matrices
            C1 = noise_variance_C(processer.S_mu(sigma_mus[0]),processer.S_s_t(sigma_w), processer.hi_fi_intQ(sigma_mus[0],l), processer.hi_fi_vi(sigma_mus[0]),delta_t, l, sigma_mus[0])
            C2 = noise_variance_C(processer.S_mu(sigma_mus[1]),processer.S_s_t(sigma_w), processer.hi_fi_intQ(sigma_mus[1],l), processer.hi_fi_vi(sigma_mus[1]),delta_t, l, sigma_mus[1])

            A1 = transition_matrix(processer.mean_s_t(), delta_t, l)

            C, A = np.zeros((10,10)), np.zeros((10,10))
            C[:5,:5] = C1
            C[5:,5:] = C2
            A[:5,:5] = A1
            A[5:,5:] = A1

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

def inference_filtering(num_particles = 100, datapath = 'experiments/data/x_ns.npz'):
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    y_ns = data_read['y'][:,0] # observation, size (N, 1)
    l = data_read['l']
    c = data_read['c']
    delta_t = data_read['delta_t']
    sigma_w = data_read['sigma_w']
    sigma_mu = data_read['sigma_mu']
    alpha = data_read['alpha']
    k_v = data_read['k_v']

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez('experiments/data/filter_res.npz',n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    return average, std3, x_dashed, xs, fxs

def inference_filtering2d(num_particles = 200, datapath = 'experiments/data/2d/x_ns.npz'):
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    y_ns = data_read['y'][:,(0,2)] # observation, size (N, 1)
    l = data_read['l']
    c = data_read['c']
    delta_t = data_read['delta_t']
    sigma_w = data_read['sigma_w']
    sigma_mus = data_read['sigma_mu']
    alpha = data_read['alpha']
    k_v = data_read['k_v']

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_2d(y_ns, num_particles, c, delta_t, sigma_mus, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(f'experiments/data/2d/filter_res_{int(alpha*10)}.npz',n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    return average, std3, x_dashed, xs, fxs

def plot_result_from_stored(datapath = 'experiments/data/2d/x_ns.npz', resultpath = 'experiments/data/2d/filter_res.npz'):
    res = np.load(resultpath)
    
    data_read = np.load(datapath)
    x_dashed = data_read['x'] # true, uncentered, size (N, 5)
    c = data_read['c']
    sigma_w = data_read['sigma_w']
    alpha = data_read['alpha']
    
    average, std3, _ ,xs, fxs = process_filter_results(res['n_mus'], res['n_vars'], res['n_log_ws'], res['E_ns'], sigma_w )
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('displacement')
    pred_xs = average[:,:2] - alpha/(alpha-1) * c**(1-1/alpha)*average[:,2:4]
    plt.plot(pred_xs[:,0])
    plt.plot(x_dashed[:,0]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,2])
    plt.legend(['pred','true'])
    plt.subplot(2,1,2)
    plt.ylabel('velocity')
    plt.plot(pred_xs[:,1])
    plt.plot(x_dashed[:,1]- alpha/(alpha-1) * c**(1-1/alpha)*x_dashed[:,3])
    plt.savefig(f'experiments/figure/2d/filter/xs_{int(alpha*10)}.png')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('integral of mu')
    plt.plot(average[:,2])
    plt.plot(x_dashed[:,2])
    plt.legend(['pred','true'])
    plt.ylim([min(average[25:,2] - std3[25:,2]),max(average[25:,2] + std3[25:,2])])
    plt.fill_between(range(len(average)), average[:,2] - std3[:,2], average[:,2] + std3[:,2],
                 color='gray', alpha=0.2)
    plt.subplot(2,1,2)
    plt.ylabel('mu')
    plt.ylim([min(average[25:,-1] - std3[25:,-1]),max(average[25:,-1] + std3[25:,-1])])
    plt.fill_between(range(len(average)), average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.plot(average[:,-1])
    plt.plot(x_dashed[:,-1])
    plt.savefig(f'experiments/figure/2d/filter/mus_{int(alpha*10)}.png')
    
    plt.figure()
    plt.plot(xs*1e3, fxs)
    plt.axvline(x = sigma_w**2*1e3, color = 'g', label = '$\sigma_W^2$',linestyle='dashed',)
    plt.xlabel(r'$\sigma_W^2 / 10^{-3}$')
    plt.legend(['Posterior','True $\sigma_W^2$'])
    plt.savefig(f'experiments/figure/2d/filter/sigma_{int(alpha*10)}.png')
    # plt.show()
    

def inf_1d_fish(num_particles = 100, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -0.05
    c = 10
    N = 500
    delta_t = 1
    sigma_w = 0.1
    sigma_mu = 1e-3
    alpha = 1.5
    k_v = 10 # 1.5e4 for alpha=0.9

    x_ns = extract_track(datapath)[0,:N,0]
    # add noise
    y_ns = x_ns + +np.random.normal(0, sigma_w*k_v, N)
    num_particles = 200

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish1d",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    plt.figure()
    plt.subplot(2,1,1)
    plt.ylabel('displacement')
    pred_xs = average[:,:2] - alpha/(alpha-1) * c**(1-1/alpha)*average[:,2:4]
    plt.plot(pred_xs[:,0])
    plt.plot(x_ns, linestyle = '--', color = 'red')
    plt.scatter(range(N),y_ns,color='orange',s=5)
    plt.ylim([18,28])
    plt.legend(['pred','true','noisy'])
    plt.subplot(2,1,2)
    plt.ylabel('mu')
    plt.plot(pred_xs[:,-1])
    plt.ylim([min(pred_xs[25:,-1]),max(pred_xs[25:,-1])])
    plt.savefig(f'experiments/figure/real_data/xs_{int(alpha*10)}.png')
    
if __name__=='__main__':
    # inference_filtering2d(num_particles = 100, datapath = 'experiments/data/2d/x_ns9.npz') 
    # plot_result_from_stored()

    inf_1d_fish()

    
    
    