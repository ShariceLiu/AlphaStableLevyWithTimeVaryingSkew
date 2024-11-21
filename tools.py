import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy

def eAt(l, t):
    # Langevin model, return e^At, l is usually negative
    f1 = np.exp(l*t)/l-1/l
    f2 = np.exp(t*l)
    return np.array([[1,f1],[0, f2]])

def eAdasht(l, t):
    # langevin, A dash is transition matrix including integral of mu
    eAdasht = np.zeros((5,5))
    eAdasht[:2,:2] = eAt(l, t)
    eAdasht[2:4,2:4] = eAt(l, t)
    eAdasht[-1,-1] = 1
    
    eAdasht[2,-1] = (np.exp(l*t)-1-l*t)/l**2
    eAdasht[3, -1] = (np.exp(l*t)-1)/l
    
    return eAdasht

def Adash(l):
    # return A dash
    Adash = np.zeros((5,5))
    A = np.array([[0,1],[0,l]])
    Adash[:2,:2] = A
    Adash[2:4,2:4]= A
    Adash[3,-1]=1
    print(Adash)
    
    return Adash

def int_exp_lambda_dt(l, delta_t):
    return (np.exp(l*delta_t)-1)/l

def int_t_exp_lambda_dt(l, delta_t):
    return delta_t*np.exp(l*delta_t)/l - int_exp_lambda_dt(l, delta_t)/l

def int_ft(l, delta_t):
    """return integral of f(t) = e^{\lambda t}h over s to t

    Args:
        l (float): lambda
        delta_t (float): t-s

    Returns:
        ndarray: corresponding to the transpose of result
    """
    term2 = int_exp_lambda_dt(l, delta_t)
    term1 = (term2- delta_t)/l
    return np.array([term1, term2])

def int_qt(l, delta_t):
    """return a 1d array, 
    qt is the term from the exp A dash"""
    term2 = (int_exp_lambda_dt(l, delta_t)-delta_t)/l
    term1 = (term2- delta_t**2/2)/l
    return np.array([term1, term2])

def int_qq_T(l, delta_t):
    """return a 1d array, 
    qt is the term from the exp A dash,
    returns integral of q*q.T"""
    term22 = int_exp_lambda_dt(2*l, delta_t)- 2*int_exp_lambda_dt(l, delta_t)+delta_t
    term21 = (term22 - l*int_t_exp_lambda_dt(l, delta_t) + l*delta_t**2/2)/l
    term11 = (term22 - 2*l*int_t_exp_lambda_dt(l, delta_t) + l*delta_t**2)/l**2 + delta_t**3/3
    return np.array([[term11, term21],[term21, term22]])/l**2

class alphaStableJumpsProcesser():
    """
    A class to process jumps
    """
    def __init__(self, gammas, vs,  alpha, delta_t, l=0):
        self.gammas = gammas
        self.vs = vs
        self.alpha = alpha
        self.delta_t = delta_t

        N = len(gammas)
        fVs = np.zeros((N, 2)) # f_t(V_i), row vectors
        hGammas = gammas**(-1/alpha) # jump size from h(gamma)
        if l:
            for i in range(N):
                fVs[i,:] = eAt(l, (delta_t - vs[i]))@np.array([0,1])
        else: #l=0, pure noise
            fVs[:,0] += 1
            
        self.fVs = fVs
        self.hGammas = hGammas
        
    def S_mu(self, sigma_mu):
        """return variance caused by sum of jumps
        S_\mu = dt^(2/\alpha) \sigma_\mu^2 \sum_i\sum_j h(\Gamma_i)h(\Gamma_j)f(V_i)f^T(V_j) min(V_i, V_j)
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
                
        return sigma_mu**2*Smu* self.delta_t**(2/self.alpha)
    
    def hi_fi_vi(self, sigma_mu):
        """
        return sum h(Gamma_i)f(V_i)V_i sigma_mu^2
        """
        sum_hfv= 0
        for i in range(len(self.gammas)):
            sum_hfv+= self.hGammas[i]*self.fVs[i,:]*self.vs[i]
            
        return sum_hfv*self.delta_t**(1/self.alpha)*sigma_mu**2
    
    def hi_fi_intQ(self, sigma_mu, l):
        """
        return sigma_mu^2 sum h(Gamma_i)f(V_i) int_0^{V_i} Q(dt - u)du
        """
        sum_hfQ = 0
        for i in range(len(self.gammas)):
            # int_0^{V_i} = int_0^{dt} - int_0^{dt-V_i}
            sum_hfQ+= self.hGammas[i]*np.outer(self.fVs[i,:],(int_qt(l, self.delta_t) - int_qt(l, self.delta_t-self.vs[i])))
            
        return sum_hfQ*self.delta_t**(1/self.alpha)*sigma_mu**2
    
    def mean_s_t(self, mus = 1):
        """reurn row vectors, if mus = None, consider mu = 1"""
        if type(mus) is int:
            mus = [mus]*len(self.gammas)
        mean_st= 0
        for i in range(len(self.gammas)):
            mean_st+= self.hGammas[i]*self.fVs[i,:]*mus[i]
            
        return mean_st*self.delta_t**(1/self.alpha)
    
    def S_s_t(self, sigma_W):
        S_st = 0
        for i in range(len(self.gammas)):
            S_st+= (self.hGammas[i])**2*np.outer(self.fVs[i,:],self.fVs[i,:])
            
        return S_st*sigma_W**2*self.delta_t**(2/self.alpha)
    
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
        dt_i = T-v0
        mus[-1] = np.random.normal(u1, sigma_mu*np.sqrt(dt_i))
        return np.array(vs), np.array(gammas), mus
    else:
        return np.array(vs), np.array(gammas)
    

def transition_matrix(sum_hGamma_ft, delta_t, l, with_int = True):
    if with_int:
        matrix = eAdasht(l, delta_t)
        matrix[:2, -1] = sum_hGamma_ft
    else:
        matrix = np.zeros((3,3))
        matrix[:2,:2] = eAt(l, delta_t)
        matrix[:2,-1] = sum_hGamma_ft
        matrix[-1,-1] = 1
    return matrix

def noise_variance_C(S_mu, S_st, hi_fi_intQ, hi_fi_vi, delta_t, l, sigma_mu, with_int = True):
    if with_int:
        C = np.zeros((5,5))
        C[:2,:2] = S_mu + S_st
        C[2:4,2:4] = int_qq_T(l, delta_t)*sigma_mu**2
        C[2:4,-1] = int_qt(l, delta_t)*sigma_mu**2
        C[-1,2:4] = C[2:4,-1]
        C[-1,-1] = delta_t*sigma_mu**2
        C[:2,2:4] = hi_fi_intQ
        C[2:4,:2] = hi_fi_intQ.T
        C[:2,-1] = hi_fi_vi
        C[-1,:2] = hi_fi_vi
        
    else:
        C = np.zeros((3,3))
        C[:2,:2] = S_mu + S_st
        C[-1,-1] = delta_t*sigma_mu**2
        C[:2,-1] = hi_fi_vi
        C[-1,:2] = hi_fi_vi
    return C