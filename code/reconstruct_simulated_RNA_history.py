import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy.stats import poisson
from functools import lru_cache
from scipy.sparse.linalg import expm_multiply, eigs
from scipy.sparse import coo_matrix, csr_matrix, diags

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

def stationary_from_params(alpha, beta, gamma, states):
    mean_u = alpha / beta
    mean_s = alpha / gamma

    # unpack states into arrays
    u_vals = np.array([u for (u, s) in states], dtype=int)
    s_vals = np.array([s for (u, s) in states], dtype=int)

    # Poisson pmf for each state coordinate
    p_u = poisson.pmf(u_vals, mean_u)
    p_s = poisson.pmf(s_vals, mean_s)

    pi = p_u * p_s  # independent U and S
    pi /= pi.sum()  # renormalize on truncated grid

    return pi

def reverse_generator(A, mu):
    ## Some refs for getting reverse time Markov generator:
    ## https://arxiv.org/abs/2502.19183 (p. 3)
    ## https://www.randomservices.org/random/markov/TimeReversal2.html
    ## https://link.springer.com/book/10.1007/978-1-4612-3038-0 (p. 239)

    diag_mu = np.diag(mu)
    diag_inv_mu = np.diag(1.0/mu)
    A_rev_off = diag_inv_mu @ A.T @ diag_mu
    np.fill_diagonal(A_rev_off, 0.0)
    A_rev = A_rev_off.copy()
    A_rev[np.diag_indices_from(A_rev)] = -A_rev_off.sum(axis=1) # Diagonals must be -sum(row)
    return A_rev
    
A_gene = None
X_fwd_gene = None

@lru_cache(maxsize=None)
def get_expm(state_idx, dt):
    """
    Cached expm(A_gene[state_idx] * dt).
    """
    A_k = A_gene[state_idx]
    return la.expm(A_k * dt)

@lru_cache(maxsize=None)
def get_expm_rev(state_idx, k, dt):
    """
    Cached expm(A_rev(state_idx, k) * dt) for BACKWARD direction.
    """
    A_rev = get_A_rev(state_idx, k)
    return la.expm(A_rev * dt)

@lru_cache(maxsize=None)
def get_A_rev(state_idx, k):
    """
    Cached reverse generator A_rev for BACKWARD direction at time index k.
    """
    mu_k = X_fwd_gene[:, k]
    return reverse_generator(A_gene[state_idx], mu_k)

def forward_distribution(A, Y, pi, states, t, tau, state_grid):
    global A_gene, X_fwd_gene
    A_gene = A 
    
    # For each time step, calculate next state using current state
    
    # Start with stationary distribution at t < 0 (system starts in steady state)
    X_fwd = np.zeros(shape=(len(states), len(t)))
    dt = np.mean(np.diff(t))
    X_fwd[:, 0] = expm_multiply(A[0].T * dt, pi)
    
    for k in range(0, len(t)-1): 
        t_curr, t_next = t[k], t[k+1]
        state_curr, state_next = state_grid[k], state_grid[k+1]
        x_curr = X_fwd[:, k]
       
        if state_curr == state_next:
            dt = t_next - t_curr
            # A_k = A[state_curr]
            # x_next = x_curr @ sp.linalg.expm(A_k * dt) 
            M = get_expm(state_curr, dt)
            x_next = x_curr @ M 
        
        else:   
            # State switch happens in current interval
            t_s = tau[state_next]

            # Split forward march into 2 steps
            dt1 = t_s - t_curr # left interval: [t_k, state_switch_time)
            M1 = get_expm(state_curr,  dt1)
            x_mid = x_curr @ M1  
      
            dt2 = t_next - t_s # right interval: [state_switch_time, t_{k+1})
            M2 = get_expm(state_next, dt2)
            x_next = x_mid @ M2
        
        X_fwd[:, k+1] = x_next
        
        print("Jump fowwards from t =", np.round(t_curr, 3), "to t =", np.round(t_next, 3), ":")
        print("Highest probability state (# unspliced, spliced):", states[np.argmax(x_curr)])
        idx = np.searchsorted(t, t_next)
        print("Simulated (# unspliced, spliced):", np.round(Y[idx, :, 0], 2), np.round(Y[idx, :, 1], 2))
        print("Sum X(t_k) =", np.sum(x_curr))
        print("-------") 
    
    X_fwd_gene = X_fwd
    return X_fwd

def backward_distribution(Y, states, index_for, t, tau, state_grid):
    # Y: U and S count matrices

    global A_gene, X_fwd_gene
    
    n_cells = Y.shape[0]

    # Initialize backwards trajectory with observed counts for each cell
    U_curr, S_curr = np.round(Y[:, 0]), np.round(Y[:, 1])
    X_curr = np.zeros(shape=(len(states), n_cells), dtype="float")
    for cell_idx in range(n_cells):
        X_curr[index_for[(U_curr[cell_idx], S_curr[cell_idx])], cell_idx] = 1.0 
        
    X_bw = np.zeros(shape=(len(states), len(t), Y.shape[0])) 

    # Start backwards trajectory at cell's inferred position in time
    t_obs = np.asarray([np.searchsorted(t, i) for i in t])

    for k in reversed(range(1, len(t))):
        t_prev, t_curr = t[k-1], t[k]
        state_prev, state_curr = state_grid[k-1], state_grid[k]

        # Only calculate backwards trajectory for cells at t = t[k] or later
        mask = t_obs >= k
        X_curr_block = X_curr[:, mask]
        
        if state_prev == state_curr:
            dt = t_curr - t_prev
            M_rev = get_expm_rev(state_curr, k, dt)
            X_prev = (X_curr_block.T @ M_rev).T
            
        else:
            # State switch happens in current interval             
            t_s = tau[state_curr]

            # Split backward march into 2 steps
            dt2 = t_curr - t_s # right interval: (state_switch_time, t_k]
            M2_rev = get_expm_rev(state_curr, k, dt2)
            X_mid = (X_curr_block.T @ M2_rev)
            
            dt1 = t_s - t_prev # left interval: (t_{k-1}, state_switch_time]
            M1_rev = get_expm_rev(state_prev, k, dt1)
            X_prev = (X_mid @ M1_rev).T

        X_bw[:, k, mask] = X_curr_block
        X_curr[:, mask] = X_prev
        
    X_bw[:, k-1, mask] = X_prev

    # Cells at t=0 only have observed data
    mask = t_obs == 0   
    X_bw[:, 0, mask] = X_curr[:, mask]
        
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

################################################################################

def create_transition_matrix_sparse(reactions, states, index_for, U_max, S_max):
    data = []
    rows = []
    cols = []

    for i, (u, s) in enumerate(states):
        out_rate = 0.0
        for rxn in reactions:
            u2 = u + rxn.dU
            s2 = s + rxn.dS
            
            if (u2 < 0) or (u2 > U_max) or (s2 < 0) or (s2 > S_max):
                continue

            rate = rxn.rate_fn(u, s)
            j = index_for[(u2, s2)]
            rows.append(i)
            cols.append(j)
            data.append(rate)
            out_rate += rate

        # diagonal
        rows.append(i)
        cols.append(i)
        data.append(-out_rate)

    n_states = len(states)
    A_sparse = csr_matrix((data, (rows, cols)), shape=(n_states, n_states))
    return A_sparse

def reverse_generator_sparse(A, mu):
    ## Some refs for getting reverse time Markov generator:
    ## https://arxiv.org/abs/2502.19183 (p. 3)
    ## https://www.randomservices.org/random/markov/TimeReversal2.html
    ## https://link.springer.com/book/10.1007/978-1-4612-3038-0 (p. 239)
 
    # # Work with A^T in COO format (easy access to row/col/data)
    # A_T = A.T.tocoo()

    # # For A_T (i,j) = A[j,i]
    # # Want A_rev_off[i,j] = A[j,i] * mu[j] / mu[i] for i != j
    # data = A_T.data * mu[A_T.row] / mu[A_T.col]

    # # Remove diagonal entries (i == j)
    # mask = A_T.row != A_T.col
    # row_off = A_T.row[mask]
    # col_off = A_T.col[mask]
    # data_off = data[mask]

    # n = A.shape[0]
    # A_rev_off = coo_matrix((data_off, (row_off, col_off)), shape=(n, n)).tocsr()

    # # Row sums for diagonal (ensure rows sum to 0)
    # row_sums = np.array(A_rev_off.sum(axis=1)).ravel()
    # A_rev = A_rev_off - diags(row_sums)
    
    # A_fwd: sparse (n×n), mu: (n,)
    mu_safe = np.maximum(mu, 1e-12)
    ratio = mu_safe[None, :] / mu_safe[:, None]          # dense (n×n)
    A_rev_off = (A.T.multiply(ratio)).tolil()
    A_rev_off.setdiag(0.0)
    diag = -np.array(A_rev_off.sum(axis=1)).ravel()
    A_rev_off.setdiag(diag)
    A_rev = A_rev_off.tocsr()


    return A_rev

def forward_distribution_blocked_sparse(A, pi, states, t, tau, state_grid):
    global A_gene, X_fwd_gene
    A_gene = A 
    
    # For each time step, calculate next state using current state
    
    n_states = len(states)
    n_t = len(t)
    
    # Start with stationary distribution at t < 0 (system starts in steady state)
    X_fwd = np.zeros(shape=(n_states, len(t)))
    dt = np.mean(np.diff(t)) 
    X_fwd[:, 0] = expm_multiply(A[0].T * dt, pi)

    k = 0
    while k < n_t - 1:
        s = state_grid[k]

        # ---- find maximal constant-state block starting at k ----
        k_block_end = k
        while (
            k_block_end < n_t - 1
            and state_grid[k_block_end] == state_grid[k_block_end + 1]
        ):
            k_block_end += 1

        if k_block_end > k:
            # ---- constant generator A[s] on [t_k, ..., t_{k_block_end}] ----
            A_s_T = A[s].T

            # total time span and number of timepoints in this block
            t_start = t[k]
            t_end = t[k_block_end]
            num = k_block_end - k + 1  # including the starting point

            # evolve from X_fwd[:, k] over this whole block
            # expm_multiply will compute e^{A_s_T * tau_j} X_fwd[:,k] for
            # tau_j linearly spaced between 0 and (t_end - t_start)
            block = expm_multiply(
                A_s_T,
                X_fwd[:, k],
                start=0.0,
                stop=t_end - t_start,
                num=num,
            )
            # block.shape == (num, n_states); we want columns = time
            X_fwd[:, k : k_block_end + 1] = block.T

            k = k_block_end  # jump to the end of the block

        else:
            # ---- single step where state changes within this interval ----
            t_curr, t_next = t[k], t[k+1]
            state_curr, state_next = state_grid[k], state_grid[k+1]
            x_curr = X_fwd[:, k]

            if state_curr == state_next:
                # (shouldn't really happen here, but keep for safety)
                dt = t_next - t_curr
                x_next = expm_multiply(A[state_curr].T * dt, x_curr)
            else:
                # State switch happens somewhere inside (t_curr, t_next]
                t_s = tau[state_next]

                # left sub-interval: [t_curr, t_s)
                dt1 = t_s - t_curr
                x_mid = expm_multiply(A[state_curr].T * dt1, x_curr)

                # right sub-interval: [t_s, t_next]
                dt2 = t_next - t_s
                x_next = expm_multiply(A[state_next].T * dt2, x_mid)

            X_fwd[:, k+1] = x_next
            k += 1

    X_fwd_gene = X_fwd
    return X_fwd

@lru_cache(maxsize=None)
def get_A_rev_sparse(state_idx, k):
    mu_k = X_fwd_gene[:, k]
    return reverse_generator_sparse(A_gene[state_idx], mu_k)

def backward_distribution_sparse(Y, states, index_for, t, tau, state_grid):
    # Y: U and S count matrices
    # Q: posterior probability of each cell (shape: # cells x len(t))

    global A_gene, X_fwd_gene
    
    n_cells = Y.shape[0]

    # Initialize backwards trajectory with observed counts
    U_curr, S_curr = np.round(Y[:, 0]), np.round(Y[:, 1])
    X_curr = np.zeros(shape=(len(states), n_cells), dtype="float")
    for cell_idx in range(n_cells):
        X_curr[index_for[(U_curr[cell_idx], S_curr[cell_idx])], cell_idx] = 1.0 
            
    X_bw = np.zeros(shape=(len(states), len(t), Y.shape[0])) 

    # Start backwards trajectory at cell's inferred position in time
    t_obs = np.asarray([np.searchsorted(t, i) for i in t])

    for k in reversed(range(1, len(t))):
        t_prev, t_curr = t[k-1], t[k]
        state_prev, state_curr = state_grid[k-1], state_grid[k]
        
        # Only calculate backwards trajectory for cells at t = t[k] or later
        mask = t_obs >= k
        X_curr_block = X_curr[:, mask]

        if state_prev == state_curr:
            dt = t_curr - t_prev
            A_rev = get_A_rev_sparse(state_curr, k)
            X_prev = expm_multiply(A_rev.T * dt, X_curr_block) 
            
        else:
            # State switch happens in current interval             
            t_s = tau[state_curr]

            # Split backward march into 2 steps
            dt2 = t_curr - t_s # right interval: (state_switch_time, t_k]
            A_rev2 = get_A_rev_sparse(state_curr, k)
            X_mid = expm_multiply(A_rev2.T * dt2, X_curr_block) 
        
            dt1 = t_s - t_prev # left interval: (t_{k-1}, state_switch_time]
            A_rev1 = get_A_rev_sparse(state_prev, k) 
            X_prev = expm_multiply(A_rev1.T * dt1, X_mid) 

        X_bw[:, k, mask] = X_curr_block
        X_curr[:, mask] = X_prev
    
    X_bw[:, k-1, mask] = X_prev

    # Cells at t=0 only have observed data
    mask = t_obs == 0   
    X_bw[:, 0, mask] = X_curr[:, mask]
    
    return X_bw

################################################################################

# def stationary_from_transition_matrix(A): 
#     w, v = np.linalg.eig(A.T)
#     idx = np.argmin(np.abs(w)) # Eigenvector corresponding to eigenvalue closest to 0
#     pi = np.real(v[:, idx])
#     if pi.sum() < 0:
#         pi = -pi
#     pi[pi < 0] = 0.0
#     pi /= pi.sum()
#     return pi


# def backward_distribution(Y, X_bw, states, t, tau, state_grid):
#     global A_gene, X_fwd_gene

#     # For each time step, calculate previous state using current state

#     for k in reversed(range(1, len(t))):
#         t_prev, t_curr = t[k-1], t[k]
#         state_prev, state_curr = state_grid[k-1], state_grid[k]
#         x_curr = X_bw[:, k]
            
#         if state_prev == state_curr:
#             dt = t_curr - t_prev
#             M_rev = get_expm_rev(state_curr, k, dt)
#             x_prev = x_curr @ M_rev
        
#         else:
#             # State switch happens in current interval             
#             t_s = tau[state_curr]

#             # Split backward march into 2 steps
#             dt2 = t_curr - t_s # right interval: (state_switch_time, t_k]
#             M2_rev = get_expm_rev(state_curr, k, dt2)
#             x_mid = x_curr @ M2_rev 
            
#             dt1 = t_s - t_prev # left interval: (t_{k-1}, state_switch_time]
#             M1_rev = get_expm_rev(state_prev, k, dt1)
#             x_prev = x_mid @ M1_rev  

#         X_bw[:, k-1] = x_prev
        
#         ###########################################################################
       
#         print("Jump backwards from t =", np.round(t_curr, 3), "to t =", np.round(t_prev, 3), ":")
#         print("Highest probability state (# unspliced, spliced):", states[np.argmax(x_curr)])
#         idx = np.searchsorted(t, t_prev)
#         print("Simulated (# unspliced, spliced):", np.round(Y[idx, :, 0], 2), np.round(Y[idx, :, 1], 2))
#         print("Sum X(t_k) =", np.sum(x_curr))
#         print("-------") 
        
#     return X_bw
