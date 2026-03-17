import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


import numpy as np


def simulate_system(
    W_e,
    W_u,
    Gamma_H,
    Gamma_B,
    A,
    A_tilde,
    grad_loss_array,
    x
):
    """
    Simulates the forward-backward triangular system.

    Returns:
        H : array (num_s, K)
        B : array (num_s, K)
    """
    num_s, K, _ = Gamma_H.shape
    ds = 1.0 / (num_s - 1)

    # Storage
    H = np.zeros((num_s, K))
    B = np.zeros((num_s, K))

    # ----- k = 0 base case -----

    H[:, 0] = W_e @ x
    B[:, 0] = W_u @ grad_loss_array[0]

    # ----- inductive construction -----

    for k in range(1, K):

        # ---- Forward solve H_k ----

        H[0, k] = W_e @ x  # initial condition

        for i in range(num_s - 1):

            # vectors of size k
            gamma_vec = Gamma_H[i, :k, k]
            B_vec = B[i, :k]

            # matrix k x k
            A_block = A[i, :k, :k]

            rhs = gamma_vec @ (A_block @ B_vec)

            H[i + 1, k] = H[i, k] + ds * rhs

        # ---- Backward solve B_k ----

        B[-1, k] = W_u @ grad_loss_array[k]  # terminal condition

        for i in reversed(range(1, num_s)):

            gamma_vec = Gamma_B[i, :k, k]
            H_vec = H[i, :k]

            A_block = A_tilde[i, :k, :k]

            rhs = - gamma_vec @ (A_block.T @ H_vec)

            B[i - 1, k] = B[i, k] - ds * rhs

    return H, B


def simulate_system_batched(
    W_e_all,          # (D, dimin) — stacked W_e @ x for all d
    W_u_all,          # (D, dimout) — stacked W_u @ grad_loss_array[k] for all d and k
    Gamma_H,
    Gamma_B,
    A,
    A_tilde,
    grad_loss_array,
    x,
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
):
    """
    Returns H_ds: (D, num_s, K)
    """
    num_s, K, _ = Gamma_H.shape
    s_grid = np.linspace(0.0, 1.0, num_s)
    D = W_e_all.shape[0]

    def make_interp(arr):
        return interp1d(s_grid, arr, axis=0, kind="linear", assume_sorted=True)

    Gamma_H_fn = make_interp(Gamma_H)
    Gamma_B_fn = make_interp(Gamma_B)
    A_fn       = make_interp(A)
    A_tilde_fn = make_interp(A_tilde)

    # H[d, s, k] and B[d, s, k]
    H = np.zeros((D, num_s, K))
    B = np.zeros((D, num_s, K))

    # k=0 base case: broadcast over d
    H[:, :, 0] = (W_e_all @ x)[:, np.newaxis]          # (D, num_s)
    for d in range(D):
        B[d, :, 0] = W_u_all[d] @ grad_loss_array[0]   # or batch if W_u shaped right

    for k in range(1, K):

        # ---- Forward: forcing term is identical for all d ----
        B_interps = [
            [interp1d(s_grid, B[d, :, j], kind="linear", assume_sorted=True)
             for j in range(k)]
            for d in range(D)
        ]

        def forward_rhs_batched(s, h_flat):
            # h_flat: (D,) — one H_k value per d
            gamma   = Gamma_H_fn(s)[:k, k]      # (k,)
            A_block = A_fn(s)[:k, :k]           # (k, k)
            # B_s: (D, k)
            B_s = np.array([[B_interps[d][j](s) for j in range(k)] for d in range(D)])
            forcing = B_s @ A_block.T @ gamma    # (D,)
            return forcing

        h0 = W_e_all @ x                         # (D,) — all initial conditions at once
        sol_H = solve_ivp(
            forward_rhs_batched,
            t_span=(0.0, 1.0),
            y0=h0,
            method=method,
            t_eval=s_grid,
            rtol=rtol,
            atol=atol,
        )
        H[:, :, k] = sol_H.y                     # (D, num_s)

        # ---- Backward: same trick ----
        H_interps = [
            [interp1d(s_grid, H[d, :, j], kind="linear", assume_sorted=True)
             for j in range(k)]
            for d in range(D)
        ]

        def backward_rhs_batched(s, b_flat):
            gamma   = Gamma_B_fn(s)[:k, k]      # (k,)
            A_block = A_tilde_fn(s)[:k, :k]     # (k, k)
            H_s = np.array([[H_interps[d][j](s) for j in range(k)] for d in range(D)])
            forcing = H_s @ A_block @ gamma      # (D,)
            return forcing

        b0 = np.array([W_u_all[d] @ grad_loss_array[k] for d in range(D)])  # (D,)
        sol_B = solve_ivp(
            backward_rhs_batched,
            t_span=(1.0, 0.0),
            y0=b0,
            method=method,
            t_eval=s_grid[::-1],
            rtol=rtol,
            atol=atol,
        )
        B[:, :, k] = sol_B.y[:, ::-1]

    return H  # (D, num_s, K)

def simulate_system_batched_optimized(
    W_e_all, W_u_all, Gamma_H, Gamma_B, A, A_tilde, grad_loss_array, x,
    method="RK45", rtol=1e-6, atol=1e-8
):
    num_s, K, _ = Gamma_H.shape
    D = W_e_all.shape[0]
    s_grid = np.linspace(0.0, 1.0, num_s)

    # Pre-interpolate global parameters
    Gamma_H_fn = interp1d(s_grid, Gamma_H, axis=0, kind="linear", assume_sorted=True)
    Gamma_B_fn = interp1d(s_grid, Gamma_B, axis=0, kind="linear", assume_sorted=True)
    A_fn       = interp1d(s_grid, A, axis=0, kind="linear", assume_sorted=True)
    A_tilde_fn = interp1d(s_grid, A_tilde, axis=0, kind="linear", assume_sorted=True)

    H = np.zeros((D, num_s, K))
    B = np.zeros((D, num_s, K))

    # Initial conditions
    H[:, :, 0] = (W_e_all @ x)[:, np.newaxis] 
    # Vectorized initialization for B at k=0
    B[:, :, 0] = (W_u_all @ grad_loss_array[0])[:, np.newaxis]

    for k in range(1, K):
        # 1. FORWARD PASS
        B_block_interp = interp1d(s_grid, B[:, :, :k], axis=1, kind="linear", assume_sorted=True)

        def forward_rhs(s, h_flat):
            # gamma: (k,), A_block: (k, k), B_s: (D, k)
            gamma   = Gamma_H_fn(s)[:k, k]
            A_block = A_fn(s)[:k, :k]
            B_s     = B_block_interp(s) 
            
            # Efficient Matrix-Vector product: (D, k) @ ((k, k) @ (k,)) -> (D,)
            return B_s @ (A_block.T @ gamma)

        h0 = W_e_all @ x
        sol_H = solve_ivp(forward_rhs, (0.0, 1.0), h0, method=method, 
                          t_eval=s_grid, rtol=rtol, atol=atol)
        H[:, :, k] = sol_H.y

        # 2. BACKWARD PASS
        H_block_interp = interp1d(s_grid, H[:, :, :k], axis=1, kind="linear", assume_sorted=True)

        def backward_rhs(s, b_flat):
            gamma   = Gamma_B_fn(s)[:k, k]
            A_block = A_tilde_fn(s)[:k, :k]
            H_s     = H_block_interp(s)
            
            return H_s @ (A_block @ gamma)

        b0 = W_u_all @ grad_loss_array[k]
        sol_B = solve_ivp(backward_rhs, (1.0, 0.0), b0, method=method, 
                          t_eval=s_grid[::-1], rtol=rtol, atol=atol)
        B[:, :, k] = sol_B.y[:, ::-1]

    return H, B