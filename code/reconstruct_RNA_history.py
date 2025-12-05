import numpy as np
import scipy as sp

class Reaction:
    def __init__(self, dU, dS, rate_fn):
        self.dU = dU
        self.dS = dS
        self.rate_fn = rate_fn  # function: input (# u, # s) -> rate

def define_reactions(alpha, beta, gamma):
    return [
        # Transcription: (u, s) -> (u+1, s)
        Reaction(dU=1, dS=0, rate_fn=lambda u, s: alpha),
        
        # Splicing: (u, s) -> (u-1, s+1)
        Reaction(dU=-1, dS=1, rate_fn=lambda u, s: beta * u),
        
        # Degradation: (u, s) -> (u, s-1)
        Reaction(dU=0, dS=-1, rate_fn=lambda u, s: gamma * s),
    ]

def enumerate_states(U_max, S_max):
    states = []
    index_for = {}
    for u in range(U_max + 1):
        for s in range(S_max + 1):
            idx = len(states)
            states.append((u, s))
            index_for[(u, s)] = idx
    return states, index_for

def create_transition_matrix(reactions, U_max, S_max):
    states, index_for = enumerate_states(U_max, S_max) 
    n_states = len(states)
    A = np.zeros((n_states, n_states))

    for i, (u, s) in enumerate(states):
        out_rate = 0.0
        
        for rxn in reactions:
            u2 = u + rxn.dU
            s2 = s + rxn.dS

            if (u2 < 0) or (u2 > U_max) or (s2 < 0) or (s2 > S_max):
                continue
                    
            rate = rxn.rate_fn(u, s)
            j = index_for[(u2, s2)]
            A[i, j] += rate
            out_rate += rate
            
        A[i, i] = -out_rate
    
    return A

def stationary_from_transition_matrix(A): 
    w, v = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(w)) # Eigenvector corresponding to eigenvalue closest to 0
    pi = np.real(v[:, idx])
    if pi.sum() < 0:
        pi = -pi
    pi[pi < 0] = 0.0
    pi /= pi.sum()
    return pi

def forward_distribution(A, pi, states, t, tau, state_grid):
    # For each time step, calculate next state using current state
    X_fwd = np.zeros(shape=(len(states), len(t)))
    X_fwd[:, 0] = pi

    for k in range(0, len(t)-1): 
        t_curr, t_next = t[k], t[k+1]
        state_curr, state_next = state_grid[k], state_grid[k+1]
        x_curr = X_fwd[:, k]
        
        if state_curr == state_next:
            dt = t_next - t_curr
            A_k = A[state_curr]
            x_next = x_curr @ sp.linalg.expm(A_k * dt) 
            
        else:   
            # State switch happens in current interval
            t_s = tau[state_next]

            # Split backward march into 2 steps
            dt1 = t_s - t_curr # left interval: [t_k, state_switch_time)
            A_k1 = A[state_curr]
            x_mid = x_curr @ sp.linalg.expm(A_k1 * dt1)
      
            dt2 = t_next - t_s # right interval: [state_switch_time, t_{k+1})
            A_k2 = A[state_next]
            x_next = x_mid @ sp.linalg.expm(A_k2 * dt2) 
        
        X_fwd[:, k+1] = x_next
        
    return X_fwd

def backward_distribution(Y, A, X_bw, X_fwd, states, t, tau, state_grid):
    ## Some refs for getting reverse time Markov generator:
    ## https://arxiv.org/abs/2502.19183 (p. 3)
    ## https://www.randomservices.org/random/markov/TimeReversal2.html
    ## https://link.springer.com/book/10.1007/978-1-4612-3038-0 (p. 239)

    def _reverse_generator(A, mu):
        diag_mu = np.diag(mu)
        diag_inv_mu = np.diag(1.0/mu)
        A_rev_off = diag_inv_mu @ A.T @ diag_mu
        np.fill_diagonal(A_rev_off, 0.0)
        A_rev = A_rev_off.copy()
        A_rev[np.diag_indices_from(A_rev)] = -A_rev_off.sum(axis=1) # Diagonals must be -sum(row)
        return A_rev
    
    ###########################################################################
    
    # For each time step, calculate previous state using current state

    for k in reversed(range(1, len(t))):
        t_prev, t_curr = t[k-1], t[k]
        state_prev, state_curr = state_grid[k-1], state_grid[k]

        x_curr = X_bw[:, k]
        mu_k = X_fwd[:, k]
            
        if state_prev == state_curr:
            dt = t_curr - t_prev
            A_k = A[state_curr]
            A_k_rev =_reverse_generator(A_k, mu_k) 
            x_prev = x_curr @ sp.linalg.expm(A_k_rev * dt)
        
        else:
            # State switch happens in current interval             
            t_s = tau[state_curr]

            # Split backward march into 2 steps
            dt2 = t_curr - t_s # right interval: (state_switch_time, t_k]
            A_k2 = A[state_curr]
            A_k2_rev =_reverse_generator(A_k2, mu_k)
            x_mid = x_curr @ sp.linalg.expm(A_k2_rev * dt2) 
            
            dt1 = t_s - t_prev # left interval: (t_{k-1}, state_switch_time]
            A_k1 = A[state_prev]
            A_k1_rev =_reverse_generator(A_k1, mu_k) 
            x_prev = x_mid @ sp.linalg.expm(A_k1_rev * dt1) 
    
        X_bw[:, k-1] = x_prev
        
        ###########################################################################
       
        print("Jump backwards from t =", np.round(t_curr, 3), "to t =", np.round(t_prev, 3), ":")
        print("Highest probability state (# unspliced, spliced):", states[np.argmax(x_curr)])
        idx = np.searchsorted(t, t_prev)
        print("Simulated (# unspliced, spliced):", np.round(Y[idx, :, 0], 2), np.round(Y[idx, :, 1], 2))
        print("Sum X(t_k) =", np.sum(x_curr))
        print("-------") 
        
    return X_bw

def marginal_distribution(X, U_max, S_max, t, states):
    X_u = np.zeros(shape=(U_max + 1, len(t)))
    X_s = np.zeros(shape=(S_max + 1, len(t)))

    for k in range(0, len(t)):
        for i in range(U_max + 1):
            s_state_indices = [idx for idx, st in enumerate(states) if st[0] == i]
            # Sum probs over all possible S corresponding to the i-th U
            X_u[i, k] = np.sum(X[s_state_indices, k])
            
        for i in range(S_max + 1):
            u_state_indices = [idx for idx, st in enumerate(states) if st[1] == i]
            X_s[i, k] = np.sum(X[u_state_indices, k])
    
    return X_u, X_s