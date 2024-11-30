import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

        # normalise weights
        n_log_ws[m,:] = n_log_ws[m,:]- np.log(sum(np.exp(n_log_ws[m,:])))

    n_vars /= sigma_w**2 # if marginalizing sigma

    return n_mus, n_vars, n_log_ws, E_ns

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
    
    if D == 3: # test if it's 2 dim
        d_a = 1
    elif D == 6:
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
    xs = np.linspace(sigma_w**2*0.8,sigma_w**2*1.3,100)
    fxs = 0
    for p in range(num_particles):
        x_beta = xs/betas[-1,p]
        pdf = invgamma.pdf(x_beta, alpha)/betas[-1,p]*np.exp(n_log_ws[-1,p])
        fxs += pdf

    return average, std3, betas, xs, fxs

def inf_1d_fish(num_particles = 100, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -1e-2 #1e-2
    c = 5
    N = 1000
    startx = 3000
    delta_t = 1
    sigma_w = 0.05
    sigma_mu = 5e-4 #1e-3
    alpha = 1.6
    k_v = 500 # 500 for x

    x_ns = extract_track(datapath)[0,startx:(startx+N),2]
    # add noise
    y_ns = x_ns #+np.random.normal(0, sigma_w*k_v, N)
    num_particles = 200

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, delta_t, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish1d_z",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    plt.ylabel('displacement')
    pred_xs = average[:,:2]
    plt.plot(pred_xs[:,0])
    plt.plot(x_ns, linestyle = '--', color = 'red')

    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(range(len(average)), average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    
    # plt.scatter(range(N),y_ns,color='orange',s=5)
    plt.legend(['pred','noisy'])

    plt.subplot(3,1,2)
    plt.ylabel('velocity')
    plt.plot(pred_xs[:,1])
    plt.plot(x_ns, linestyle = '--', color = 'red')

    plt.ylim([min(average[100:,1] - std3[100:,1]),max(average[100:,1] + std3[100:,1])])
    plt.fill_between(range(len(average)), average[:,1] - std3[:,1], average[:,1] + std3[:,1],
                 color='gray', alpha=0.2)

    plt.subplot(3,1,3)
    plt.ylabel('mu')
    plt.plot(average[:,-1])
    plt.ylim([min(average[100:,-1] - std3[100:,-1]),max(average[100:,-1] + std3[100:,-1])])
    plt.fill_between(range(len(average)), average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/fish/zs_{int(alpha*10)}_l{int(abs(l))}.png')

def inf_finance(num_particles = 200):
    datapath = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\dataEurUS.mat"

    l = -1e-2
    c = 10
    N = 500
    T = 1
    sigma_w = 0.05
    sigma_mu = 2e-3
    alpha = 1.6
    k_v = 1000

    x_ns, t_ns = extract_mat_data(datapath)
    x_ns = x_ns[:N]*5e4 - x_ns[0]*5e4
    t_ns = t_ns[:N]*5e4
    delta_ts = t_ns[1:] - t_ns[:-1]

    # add noise
    y_ns = x_ns # +np.random.normal(0, sigma_w*k_v, N)
    # plt.plot(t_ns, x_ns)
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    # plt.savefig(r'experiments\figure\real_data\finance\noisy')


    n_mus, n_vars, n_log_ws, E_ns = particle_filter_1d(y_ns, num_particles, c, T, sigma_mu, sigma_w, k_v*sigma_w, alpha, l, delta_ts)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    # np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\finance",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    plt.ylabel('displacement')
    pred_xs = average[:,:2]
    plt.plot(t_ns, pred_xs[:,0])
    plt.plot(t_ns, x_ns, linestyle = '--', color = 'red')
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    plt.ylim([min(average[25:,0] - std3[25:,0]),max(average[25:,0] + std3[25:,0])])
    plt.fill_between(t_ns, average[:,0] - std3[:,0], average[:,0] + std3[:,0],
                 color='gray', alpha=0.2)
    plt.legend(['pred','noisy'])
    
    plt.subplot(3,1,2)
    plt.ylabel('velocity')
    plt.plot(t_ns, pred_xs[:,1])
    # plt.scatter(t_ns, y_ns,color='orange',s=5)
    plt.ylim([min(average[25:,1] - std3[25:,1]),max(average[25:,1] + std3[25:,1])])
    plt.fill_between(t_ns, average[:,1] - std3[:,1], average[:,1] + std3[:,1],
                 color='gray', alpha=0.2)
    
    
    plt.subplot(3,1,3)
    plt.ylabel('mu')
    plt.plot(t_ns, average[:,-1])
    plt.ylim([min(average[25:,-1] - std3[25:,-1]),max(average[25:,-1] + std3[25:,-1])])
    # plt.ylim([-np.mean(np.abs(average[25:,-1]))*3,np.mean(np.abs(average[25:,-1]))*3])
    plt.fill_between(t_ns, average[:,-1] - std3[:,-1], average[:,-1] + std3[:,-1],
                 color='gray', alpha=0.2)
    plt.savefig(f'experiments/figure/simplified/finance/xs_{int(alpha*10)}_l{int(abs(l))}.png')

def inf_2d_fish(num_particles = 100, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -1e-2
    c = 5
    N = 1000
    startx = 3000
    delta_t = 1
    sigma_w = 0.05
    sigma_mus = [1e-3, 1e-3]
    alpha = 1.6
    k_v = 500 # 1.5e4 for alpha=0.9

    tracks = extract_track(datapath)
    y_ns = tracks[0,startx:(startx+N),[0,2]].T

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_2d(y_ns, num_particles, c, delta_t, sigma_mus, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish2d",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    m = 200
    pred_xs = average[m:,:]
    y_ns = y_ns[m:,:]
    N = N - m
    plt.ylabel('displacement')
    plt.plot(pred_xs[:,0], pred_xs[:,3], label='Pred')
    plt.plot(y_ns[:,0], y_ns[:,1], linestyle = '--', color = 'red', label='Noisy')
    quiver_idx = np.linspace(start = 0, stop = N-1, num = 50, dtype=np.int32)
    plt.quiver(pred_xs[quiver_idx,0], pred_xs[quiver_idx,3], pred_xs[quiver_idx,2], pred_xs[quiver_idx,-1], label='Mu')
    plt.plot(y_ns[0,0], y_ns[0,1], 'go', label='Start')
    plt.legend()
    plt.savefig(f'experiments/figure/simplified/fish/2d_xs_{int(alpha*10)}_l{int(abs(l))}.png')

    
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
    return

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
    
def inf_3d_fish(num_particles = 100, datapath=r'C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\fish\3DZeF20Lables\train\ZebraFish-01\gt\gt.txt'):
    l = -1e-2
    c = 5
    N = 500
    startx = 3000
    delta_t = 1
    sigma_w = 0.05
    sigma_mus = [1e-3, 1e-3, 1e-3]
    alpha = 1.6
    k_v = 500 # 1.5e4 for alpha=0.9

    tracks = extract_track(datapath)
    y_ns = tracks[0,startx:(startx+N),:]

    n_mus, n_vars, n_log_ws, E_ns = particle_filter_3d(y_ns, num_particles, c, delta_t, sigma_mus, sigma_w, k_v*sigma_w, alpha, l, delta_t)
    average, std3, _ ,xs, fxs = process_filter_results(n_mus, n_vars, n_log_ws, E_ns, sigma_w)
    np.savez(r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\alpha_stable_levy\stable_levy_code\data\real_data\infe\fish3d",n_mus=n_mus, n_vars = n_vars, n_log_ws = n_log_ws, E_ns = E_ns, allow_pickle=True)

    m = 100
    pred_xs = average[m:,:]
    y_ns = y_ns[m:,:]
    N = N - m
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pred_xs[:,0], pred_xs[:,3], pred_xs[:,6], label='Pred')
    ax.plot(y_ns[:,0], y_ns[:,1], y_ns[:,2], linestyle = '--', color = 'red', label='Noisy')
    scale = 1e3
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
    # vecfield = ax2.quiver(pred_xs[0,0], pred_xs[0,3],pred_xs[0,6], pred_xs[0,2]*scale,pred_xs[0,5]*scale, pred_xs[0,-1]*scale, label='Mu')
    # ax2.set(xlim=[min(pred_xs[:,0]) - 2.5, max(pred_xs[:,0]) + 2.5], \
    #        ylim=[min(pred_xs[:,3]) - 2.5, max(pred_xs[:,3]) + 2.5], \
    #         zlim=[min(pred_xs[:,6]) - 2.5, max(pred_xs[:,6]) + 2.5])

    def update(frame):
        # update the line plot:
        line1.set_xdata(pred_xs[:frame,0])
        line1.set_ydata(pred_xs[:frame,3])
        line1.set_3d_properties(pred_xs[:frame,6])
        line2.set_xdata(y_ns[:frame,0])
        line2.set_ydata(y_ns[:frame,1])
        line2.set_3d_properties(y_ns[:frame,2])
        # ax2.clear()
        # ax2.quiver(pred_xs[frame,0], pred_xs[frame,3],pred_xs[frame,6], pred_xs[frame,2]*scale,pred_xs[frame,5]*scale, pred_xs[frame,-1]*scale, label='Mu')
        return (line1, line2)
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=30)
    plt.show()
    # return
    ani.save(filename=f'C:/Users/95414/Desktop/CUED/phd/year1/mycode/alpha_stable_levy/stable_levy_code/data/real_data/video/2d_xs_{int(alpha*10)}_l{int(abs(l))}.gif', writer="pillow")

if __name__=='__main__':
    # inference_filtering2d(num_particles = 100, datapath = 'experiments/data/2d/x_ns9.npz') 
    # plot_result_from_stored()

    # inf_finance()
    # inf_1d_fish()
    inf_2d_fish()
    # inf_3d_fish()
