import numpy as np

def get_RNA_params(topo, p, alpha_mu=2, alpha_sd=1, 
                   beta_mu=2, beta_sd=0.5, 
                   gamma_mu=0.5, gamma_sd=0.5, random_seed=0):
    ## p: No. genes
    ## alpha_mu: Transcription rate mean for lognormal distribution
    ## beta_mu: Splicing rate mean " " "
    ## gamma_mu: RNA degradation rate mean " " "
    
    np.random.seed(random_seed)

    n_states = len(set(topo.flatten()))
    theta = np.zeros((p, n_states + 2))
    theta[:,:n_states] = np.random.lognormal(alpha_mu, alpha_sd, size=(p, n_states))
    theta[:,-2] = np.random.lognormal(beta_mu, beta_sd, size=p)
    theta[:,-1] = np.random.lognormal(gamma_mu, gamma_sd, size=p)
    theta[:,:n_states] /= theta[:,-2, None] # Normalize transcription rates by splicing rate (get_Y() assumes a = alpha/beta)
    
    return theta
        
def get_protein_params(p, transl_rate_mu=5, transl_rate_sd=1, 
                       protein_deg_rate_mu=.015, protein_deg_rate_sd=1.5, 
                       transl_rate_per_state=False, topo=None, random_seed=0):
    ## p: No. genes
    ## transl_rate_mu: Translation rate mean for lognormal distribution
    ## deg_rate_mu: Protein degradation rate mean " " "
    ## transl_rate_per_state: Generate a different translation rate for each cell state? 
    ## topo: Required if transl_rate_per_state = True 
    
    np.random.seed(random_seed)
    
    if transl_rate_per_state:
        # if rate_cor:
        # Generate transl rates for each state that are correlated
        n_states = len(set(topo.flatten()))
    else:
        n_states = 1
        
    phi = np.zeros((p, n_states + 1))
    phi[:,:n_states] = np.random.lognormal(transl_rate_mu, transl_rate_sd, size=(p, n_states)) # Translation rate per gene
    phi[:,-1] = np.random.lognormal(protein_deg_rate_mu, protein_deg_rate_sd, size=p) # Degradation rate per gene
    
    return phi

def simulate_RNA(topo, tau, theta, n, rd_mu=None, rd_var=None, random_seed=0):
    ## Source: https://github.com/pachterlab/FGP_2024
    ## theta: RNA params
    ## tau: State switching times
    ## n: No. cells
    ## p: No. genes
    ## rd_mu: Read depth mean for beta distribution
    ## rd_var: Read depth variance for beta distribution
    
    np.random.seed(random_seed)
    
    n_states = len(set(topo.flatten()))
    p = len(theta) # No. genes
    L = len(topo) # No. lineages

    Y = np.zeros((n*L, p, 2))
    true_t = []
    true_l = []
    
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-2:]), axis=1)
        t = np.sort(np.random.uniform(tau[0], tau[-1], size=n)) # Each time point is a cell! 
        Y[l*n:(l+1)*n] = get_Y(theta_l, t, tau) # Dims: cells x genes x no. RNA species
        true_t = np.append(true_t, t)
        true_l = np.append(true_l, np.full(n, l))

    if rd_mu != None:
        a = (1-rd_mu)/rd_var - rd_mu
        b = (1/rd_mu-1)*a
        rd = np.random.beta(a=a, b=b, size=n*L)
    else:
        rd_mu = 1
        rd = np.ones(n*L)
    
    theta[:,:n_states] *= rd_mu 
    Y = rd[:, None, None]*Y
    Y_observed = np.random.poisson(Y)
    
    return Y_observed, Y, theta, rd, true_t, true_l

def simulate_protein_from_RNA(Y, topo, true_t, true_l, phi, random_seed=0):
    ## phi: Protein params
    
    np.random.seed(random_seed)
    
    L = len(topo) # No. lineages
    n = Y.shape[0] // L # No. cells per lineage
    p = Y.shape[1] # No. genes
    
    y0 = Y[0, :, 1] # RNA levels at state 0
    ss_rate = phi[:,0] / phi[:,-1] # Steady-state protein production rate = transl_rate/deg_rate
    p0 = ss_rate * y0 # Initial protein abundance assuming steady-state
    
    # Protein production paramters:
    transl_rate = phi[:,0].T
    deg_rate = phi[:,-1].reshape((1, -1))
    
    P = np.zeros((n*L, p))
    
    for l in range(L):
        t_l = true_t[true_l == l]
        dt = np.diff(t_l, prepend=t_l[0]) # Time step size for each cell along the trajectory
        t_l = t_l.reshape((-1, 1)) # Time points/cells in lineage l     
        y_l = Y[l*n:(l+1)*n, :, 1] # Spliced RNAs for lineage l
        
        p_l = p0 * np.exp(-deg_rate * t_l) # Pre-existing protein that has not yet degraded

        t_diff = t_l - t_l.T # Rows = target time; columns = past times; e.g. t_diff[m, i] = time difference between t_m and t_i 
        decay_matrix = np.exp(-t_diff[:, :, None] * deg_rate) # Decay_matrix[m, i, p] = decay factor for protein abundance at t_m from RNA available at t_i for gene p
        mask = (t_diff >= 0)[:, :, None]
        mask = np.broadcast_to(mask, decay_matrix.shape)
        decay_matrix = np.where(mask, decay_matrix, 0) # Protein abundance at time t_m can't come from RNA at time t_i > t_m
        
        y_l_dt = y_l * dt[:,None] # Multiply each timepoint's RNA by its corresponding time step size (Riemann approximation)
        protein_contrib = np.einsum('mip, ip -> mp', decay_matrix, y_l_dt) # Integrate RNA counts still surviving up to each time point
        P[l*n:(l+1)*n] = p_l + transl_rate * protein_contrib # Protein abundance in each cell = pre-existing protein + newly synthesized protein
    
    P_observed = np.random.poisson(P)
        
    return P_observed, P

def get_Y(theta, t, tau):
    ## Source: https://github.com/pachterlab/FGP_2024

    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    
    # global parameters: upper and lower limits for numerical stability
    eps = 1e-10
    
    p = len(theta)
    K = len(tau)-1 # number of states
    assert np.shape(theta)[1]==K+3
    a = theta[:,1:(K+1)].T
    beta = theta[:,-2].reshape((1,-1))
    gamma = theta[:,-1].reshape((1,-1))

    y1_0 = theta[:,0].reshape((1,-1))

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = d*y1_0
    a_ = d*a
    t = t.reshape(-1,1)
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t@beta)
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[None,k-1] * (np.exp(- (I[k,:,None] *(t-tau[k]))@beta)- np.exp(-(I[k,:,None]*(t-tau[k-1]))@beta )) \
          + a[None,k-1] * (1 - np.exp(- (idx[:,None] *(t-tau[k-1]))@beta ) )
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    # mature + c * nascent 
    y=y_0*np.exp(-t@gamma)    
    for k in range(1,K+1):
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y = y + a_[None,k-1] * (np.exp(-(I[k,:,None] * (t-tau[k]))@gamma)- np.exp(-(I[k,:,None] * (t-tau[k-1]))@gamma )) \
          +  a_[None,k-1] * (1 - np.exp(-(idx[:,None]*(t-tau[k-1]))@gamma) )

    Y = np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y