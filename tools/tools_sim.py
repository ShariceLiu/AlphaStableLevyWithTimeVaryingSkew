import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy

def eAt(l, t):
    # Langevin model, return e^At, l is usually negative
    f1 = np.exp(l*t)/l-1/l
    f2 = np.exp(t*l)
    return np.array([[1,f1],[0, f2]])

def int_exp_lambda_dt(l, dt, dv):
    """
    return \int_s^v e^{l(t-u)} du
    
    Args:
        dt = t-s
        dv = t-v
    """
    return (np.exp(l*dt)-np.exp(l*dv))/l

def int_t_exp_lambda_dt(l, dt, dv):
    """
    return \int_s^v (u-s)e^{l*(t-u)}du

    Args:
        dt = t-s
        dv = v-s
    """
    return (np.exp(l*dt) - (dv*l+1)*np.exp(l*(dt-dv)))/l**2

def int_t2_exp_lam_t(l, dt):
    """
    return \int_s^t (u-s)(t-u)e^{l*(t-u)}du

    Args:
        dt = t-s
    """
    return (l*dt * np.exp(l*dt)-2*np.exp(l*dt)+2+dt*l)/l**3

def int_ft(l, delta_t):
    """return \int_s^t e^{A(t-u)}h dt

    Args:
        l (float): lambda
        delta_t (float): t-s

    Returns:
        ndarray: corresponding to the transpose of result
    """
    term2 = int_exp_lambda_dt(l, dt=delta_t, dv= 0.0)
    term1 = (term2- delta_t)/l
    return np.array([term1, term2])

def int_ft_t(l, dt, dv):
    """
    return \int_s^v f_t(u)(u-s)du

    Args:
        dt = t-s
        dv = v-s
    """
    term2 = int_t_exp_lambda_dt(l,dt, dv)
    term1 = (term2 - 0.5*dv**2)/l
    return np.array([term1, term2])

def int_fft(l, delta_t):
    """
    return \int_s^t f_t(u)f_t(u)^T du
    """
    term22 = int_exp_lambda_dt(2*l, dt=delta_t, dv=0.0)
    term12 = (term22 - int_exp_lambda_dt(l, dt=delta_t, dv=0.0))/l
    term11 = (term22 - 2*int_exp_lambda_dt(l, delta_t, 0.0)+ delta_t)/l**2
    return np.array([[term11, term12],[term12, term22]])

def int_fft_u_s(l, delta_t): # TODO: recalculate this
    """
    return \int_s^t \int_u^t f_t(u) f_t(v)^T (u-s)dudv + its transpose
    """
    
    term4 = (int_t_exp_lambda_dt(2*l,delta_t, delta_t)- int_t_exp_lambda_dt(l, delta_t, delta_t))/l
    term3 = term4/l - int_t2_exp_lam_t(l,delta_t)/l
    term2 = (int_t_exp_lambda_dt(2*l,delta_t, delta_t)- 2*int_t_exp_lambda_dt(l, delta_t, delta_t)+0.5* delta_t**2)/l**2
    # term1 = term2/l + (delta_t**3/6 - int_t2_exp_lam_t(l, delta_t))/l**2
    term1 = (int_t_exp_lambda_dt(2*l,delta_t, delta_t) - 2*int_t_exp_lambda_dt(l, delta_t, delta_t) - l*int_t2_exp_lam_t(l, delta_t) + delta_t**2/2+l*delta_t**3/6)/l**3
    return np.array([[term1*2, term2+term3],[term2+term3, term4]])

def M_Z_2(e, alpha):
    return alpha*e**(-alpha+2)/(2-alpha)

def Sigma_E_s_t(l, delta_t, c, sigma_W, T):
    """
    epsilon=delta*c/T
    return v(s,t) M_Z^2
    """
    return int_fft(l, delta_t)*sigma_W^2/T* M_Z_2(delta_t*c/T)

def C_z(e, T, alpha):
    """
    return T^{-1} \int_0^e zQ_Z(dz)
    """
    # return alpha/(1-alpha) * c**(1-1/alpha)
    return alpha/(1-alpha)*e**(1-1/alpha)/T


class alphaStableJumpsProcesser():
    """
    A class to process jumps
    """
    def __init__(self, gammas, vs,  alpha, delta_t, c, T, l=0):
        self.gammas = gammas
        self.vs = vs
        self.alpha = alpha
        self.delta_t = delta_t
        self.epsilon = delta_t*c/T
        self.c = c
        self.T=T

        N = len(gammas)
        fVs = np.zeros((N, 2)) # f_t(V_i), row vectors
        hGammas = delta_t**(1/alpha) *gammas**(-1/alpha) # jump size from h(gamma)
        if l:
            for i in range(N):
                fVs[i,:] = eAt(l, (delta_t - vs[i]))@np.array([0,1])
        else: #l=0, pure noise
            fVs[:,0] += 1
            
        self.fVs = fVs
        self.hGammas = hGammas
        self.l=l

    def S_s_t(self, sigma_W):
        S_st = 0
        for i in range(len(self.gammas)):
            S_st+= (self.hGammas[i])**2*np.outer(self.fVs[i,:],self.fVs[i,:])
            
        return S_st*sigma_W**2
        
    def S_mu(self, sigma_mu):
        """return variance caused by sum of jumps
        S_\mu = \sigma_\mu^2 \sum_i\sum_j h(\Gamma_i)h(\Gamma_j)f(V_i)f^T(V_j) min(V_i, V_j)
        where h(\Gamma) = \Gamma^{-1/\alpha}

        Args:
            gammas (list of float): {\Gamma_i}
            vs (list of float): {V_i}, \in (0, delta_t)
            alpha (float): stable distribution parameter, \in (0,2), cannot be 1
            sigma_mu (float): std of the skewness \mu
        """
        Smu = 0
        N = len(self.gammas)
            
        for i in range(N):        
            for j in range(N):
                ff_ij = np.outer(self.fVs[i,:], self.fVs[j,:]) # return 2d array, correct form
                Smu += self.hGammas[i]*self.hGammas[j]*min([self.vs[i], self.vs[j]]) * ff_ij
                
        return sigma_mu**2*Smu
    
    def hi_fi_int_f(self, sigma_mu):
        """
        return \sigma_\mu^2 \sum h(\Gamma_i)f_t(V_i)(\int_s^V_i f_t^T (u-s)du + (V_i-s)\int_Vi^t f_t^T(u)du)+ its transpose
        """
        S=0
        N = len(self.gammas)

        for i in range(N):
            int_i = int_ft_t(self.l,dt=self.delta_t, dv= self.vs[i])+self.vs[i]* int_ft(self.l, delta_t=self.delta_t-self.vs[i])
            S += self.hGammas[i]*np.outer(self.fVs[i,:],int_i)

        return (S + S.T)*sigma_mu**2
    
    def hi_fi_vi(self, sigma_mu):
        """
        return sum h(Gamma_i)f(V_i)V_i sigma_mu^2
        """
        sum_hfv= 0
        for i in range(len(self.gammas)):
            sum_hfv+= self.hGammas[i]*self.fVs[i,:]*self.vs[i]
            
        return sum_hfv*sigma_mu**2
    
    def S_m_z(self, sigma_mu):
        return self.S_mu(sigma_mu) + sigma_mu**2*(C_z(self.epsilon, self.T, self.alpha))**2 * int_fft_u_s(self.l, self.delta_t) \
                    +self.hi_fi_int_f(sigma_mu)
    
    def Sigma_beta(self, sigma_beta):
        return sigma_beta**2 * int_fft_u_s(self.l, self.delta_t)
    
    def S_beta(self, sigma_beta):
        return sigma_beta**2 * int_ft_t(self.l, self.delta_t, self.delta_t)
    
    def S_zm_B(self, sigma_mu):
        return self.hi_fi_vi(sigma_mu) + sigma_mu**2 * int_ft_t(self.l, self.delta_t, self.delta_t)
    
    def A_12(self):
        """return sum hi f(V_i)"""
        mean_st = 0
        for i in range(len(self.gammas)):
            mean_st+= self.hGammas[i]*self.fVs[i,:]
            
        return mean_st + C_z(self.epsilon, self.T, self.alpha)*int_ft(self.l, self.delta_t)
    
    def A_13(self):
        return int_ft(self.l, self.delta_t)
    
    
    def int_mu(self, mus):
        N = len(self.gammas)
        all_vs = np.zeros(N+2)
        all_vs[0] = 0
        all_vs[1:-1] = self.vs
        all_vs[-1] = self.delta_t
        
        delta_vs = all_vs[1:] - all_vs[:N+1]
        int_mus = sum(mus*delta_vs)
        return int_mus
        
def generate_jumps(c, T=1.0, delta=1.0, sigma_mu = -1):
    """
    generate gammas and vs, 
    if sigma_mu not equal to None, return with mus, mus are 1 length longer, including the mu at time T
    """
    gamma = 0
    vs = []
    gammas = []
    while gamma<c:
        delta_gamma = np.random.exponential(scale = T/delta)
        gamma = gamma+delta_gamma
        v_i = np.random.uniform(0,delta)
        vs.append(v_i)
        gammas.append(gamma)
        
    gammas = [x for _, x in sorted(zip(vs, gammas), key=lambda pair: pair[0])]
    vs.sort()

    if sigma_mu>=0:
        v0 = 0
        u1 = 0
        mus = np.zeros(len(vs)+1)
        for i, (v_i, gamma) in enumerate(zip(vs, gammas)):
            dt_i = v_i - v0
            # generate u, as a brownian motion
            mus[i] = np.random.normal(u1, sigma_mu*np.sqrt(dt_i))
            u1 = mus[i]
            v0 = v_i
        # generate last u
        dt_i = delta-v0
        mus[-1] = np.random.normal(u1, sigma_mu*np.sqrt(dt_i))
        return np.array(vs), np.array(gammas), mus
    else:
        return np.array(vs), np.array(gammas)
    
def generate_mus(vs, sigma_mu, delta):
    v0 = 0
    u1 = 0
    mus = np.zeros(len(vs)+1)
    for i, v_i in enumerate(vs):
        dt_i = v_i - v0
        # generate u, as a brownian motion
        mus[i] = np.random.normal(u1, sigma_mu*np.sqrt(dt_i))
        u1 = mus[i]
        v0 = v_i
    # generate last u
    dt_i = delta-v0
    mus[-1] = np.random.normal(u1, sigma_mu*np.sqrt(dt_i))
    return mus
    

def transition_matrix(processor:alphaStableJumpsProcesser):
    """
    return matrix A, processor: jump processor
    """
    matrix = np.zeros((3,3))
    matrix[:2,:2] = eAt(processor.l, processor.delta_t)
    matrix[:2,-1] = processor.A_12()
    matrix[-1,-1] = 1
    return matrix

def noise_variance_C(processor: alphaStableJumpsProcesser, sigma_W, sigma_mu):
    C = np.zeros((3,3))
    C[:2,:2] = processor.S_s_t(sigma_W)+ processor.S_m_z(sigma_mu)+ \
        sigma_W**2*int_fft(processor.l, processor.delta_t)/processor.T*M_Z_2(processor.epsilon, processor.alpha)
    C[-1,-1] = processor.delta_t*sigma_mu**2
    C[:2,-1] = processor.S_zm_B(sigma_mu)
    C[-1,:2] = processor.S_zm_B(sigma_mu)
    return C

def transition_matrix_w_drift(processor:alphaStableJumpsProcesser):
    """
    return matrix A, with a random drift term
    """
    matrix = np.identity(4)
    matrix[:2,:2] = eAt(processor.l, processor.delta_t)
    matrix[:2,-2] = processor.A_12()
    matrix[:2,-1] = processor.A_13()
    return matrix

def noise_variance_C_w_drift(processor: alphaStableJumpsProcesser, sigma_W, sigma_mu, sigma_beta):
    C = np.zeros((4,4))
    C[:2,:2] = processor.S_s_t(sigma_W)+ processor.S_m_z(sigma_mu)+ \
        sigma_W**2*int_fft(processor.l, processor.delta_t)/processor.T*M_Z_2(processor.epsilon, processor.alpha) +\
        processor.Sigma_beta(sigma_beta)
    C[2,2] = processor.delta_t*sigma_mu**2
    C[:2,2] = processor.S_zm_B(sigma_mu)
    C[2,:2] = processor.S_zm_B(sigma_mu)
    C[:2,-1] = processor.S_beta(sigma_beta)
    C[-1,:2] = processor.S_beta(sigma_beta)
    C[-1,-1] = processor.delta_t*sigma_beta**2
    return C
