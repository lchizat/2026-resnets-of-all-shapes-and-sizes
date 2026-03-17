import numpy as np

from numpy.random import f
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

class CorrelationGenerator:
    def __init__(self, x: np.ndarray, y: np.ndarray, K=5, a=1.0, eta_u=1.0, eta_v=1.0, sigma_u=1.0, sigma_v=1.0,
                 sigma_we=1.0, sigma_wu=1.0, num_s=1000, rtol = 1e-5, atol = 1e-7):
        self.K = K
        self.a = a
        self.eta_u = eta_u
        self.eta_v = eta_v
        self.sigma_u2 = sigma_u**2
        self.sigma_v2 = sigma_v**2
        self.sigma_we2 = sigma_we**2
        self.sigma_wu2 = sigma_wu**2
        self.x = x
        self.y = y
        self.norm_x2 = np.sum(x**2)
        self.grad_loss = lambda f: f - y  # Example: MSE loss gradient with target y
        self.d_in = x.shape[0]
        self.d_out = y.shape[0]
        self.s = np.linspace(0, 1, num_s)
        self.num_s = num_s
        self.rtol = rtol
        self.atol = atol

        # Initialize arrays
        self.Gamma_H = np.zeros((num_s, K, K))
        self.Gamma_B = np.zeros((num_s, K, K))
        self.Lambda_HB = np.zeros((num_s, K, K))
        self.A = np.zeros((num_s, K, K))
        self.M_F1 = np.zeros((num_s, K, K))
        self.M_F2 = np.zeros((num_s, K, K))
        self.M_G1 = np.zeros((num_s, K, K))
        self.M_G2 = np.zeros((num_s, K, K))
        self.W_h = np.zeros((num_s, K, K))
        self.W_b = np.zeros((num_s, K, K))  
        self.xi_H_We = np.zeros((num_s, self.d_in, K))
        self.xi_H_Wu = np.zeros((num_s, self.d_out, K))
        self.xi_B_We = np.zeros((num_s, self.d_in, K))
        self.xi_B_Wu = np.zeros((num_s, self.d_out, K))
        self.f = np.zeros((K, self.d_out))
        self.grad_loss_array = np.zeros((K, self.d_out))

        self.set_base_case()

    @staticmethod
    def transpose(array):
        return array.swapaxes(-1, -2)

    @staticmethod
    def inner(array1, array2):
        return np.sum(array1 * array2)

    @property
    def A_tilde(self):
        return -self.transpose(self.A)

    def set_base_case(self) -> None:
        self.f[0] = np.zeros(self.d_out)
        grad_l_0 = self.grad_loss(self.f[0])
        self.grad_loss_array[0] = grad_l_0
        self.Gamma_H[:, 0, 0] = self.sigma_we2 * self.norm_x2
        self.Gamma_B[:, 0, 0] = self.sigma_wu2 * self.inner(grad_l_0, grad_l_0)
        self.Lambda_HB[:, 0, 0] = 0.0
        self.xi_H_We[:, :, 0] = self.sigma_we2 * self.x
        self.xi_H_Wu[:, :, 0] = np.zeros(self.d_out)
        self.xi_B_We[:, :, 0] = np.zeros(self.d_in)
        self.xi_B_Wu[:, :, 0] = self.sigma_wu2 * grad_l_0
        a1_value = -self.a**2 * (self.eta_u * self.sigma_v2 + self.eta_v * self.sigma_u2)
        self.A[:, 0, 0] = a1_value
        self.M_F1[:, 0, 0] = 0
        self.M_F2[:, 0, 0] = 1
        self.M_G1[:, 0, 0] = 1
        self.M_G2[:, 0, 0] = 0

    def compute_CH(self, k):
        CH = np.zeros((self.num_s, 2 * k, 2 * k))
        Lambda_kk = self.Lambda_HB[:, 0:k, 0:k]
        Gamma_b_kk = self.Gamma_B[:, 0:k, 0:k]
        A_k = self.A[:, :k, :k]
        WH_k = self.W_h[:, 0:k, :k]
        WB_k = self.W_b[:, 0:k, :k]

        b1 = Lambda_kk @ self.transpose(A_k)
        b2 = WH_k
        b3 = Gamma_b_kk @ self.transpose(A_k) + WB_k

        CH[:, :k, :k] = b1
        CH[:, :k, k:] = b2
        CH[:, k:, :k] = b3        
        return CH

    def compute_CB(self, k):
        CB = np.zeros((self.num_s, 2 * k, 2 * k))
        LambdaT_kk = self.transpose(self.Lambda_HB[:, 0:k, 0:k])
        Gamma_h_kk = self.Gamma_H[:, 0:k, 0:k]
        tildeA_k = self.A_tilde[:, :k, :k]
        WH_k = self.W_h[:, 0:k, :k]
        WB_k = self.W_b[:, 0:k, :k]

        b1 = LambdaT_kk @ self.transpose(tildeA_k)
        b2 = WB_k
        b3 = Gamma_h_kk @ self.transpose(tildeA_k) + WH_k

        CB[:, :k, :k] = b1
        CB[:, :k, k:] = b2
        CB[:, k:, :k] = b3        
        return CB
        
    def interpolate(self, fun):
        return interp1d(self.s, fun, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=True)

    def H_equation(self, k):
        CH = self.compute_CH(k)
        ic_Gamma = self.sigma_we2 * self.norm_x2 * np.ones(k)
        ic_Lambda = self.xi_B_We[0, :, :k].T @ self.x # (d_in, k)^T x (d_in,) = (k,)
        y_h_0 =  np.concatenate([ic_Gamma, ic_Lambda])
        y_h = self.solve_with_solve_ivp(CH, y_h_0, method='RK45', stiff=False)
        vec_GammaH_k, LambdaHB_Hk = y_h[:, :k], y_h[:, k:]
        self.Gamma_H[:, 0:k, k] = vec_GammaH_k
        self.Gamma_H[:, k, 0:k] = vec_GammaH_k
        self.Lambda_HB[:, k, 0:k] = LambdaHB_Hk

    def B_equation(self, k):
        CB = self.compute_CB(k)
        grad_k = self.grad_loss_array[k]
        grads_j = self.grad_loss_array[:k]
        inner_products = np.array([self.inner(grad_k, grads_j[j]) for j in range(k)])

        fc_Gamma = self.sigma_wu2 * inner_products
        fc_Lambda = self.xi_H_Wu[-1, :, :k].T @ grad_k # (d_out, k)^T x (d_out,) = (k,)
        y_b_f =  np.concatenate([fc_Gamma, fc_Lambda])
        y_b = self.solve_with_solve_ivp(CB, y_b_f, method='RK45', stiff=False, final_condition=True)
        vec_GammaB_k, LambdaHB_Bk = y_b[:, :k], y_b[:, k:]
        self.Gamma_B[:, 0:k, k] = vec_GammaB_k
        self.Gamma_B[:, k, 0:k] = vec_GammaB_k
        self.Lambda_HB[:, 0:k, k] = LambdaHB_Bk

    def xi_H_equation(self, k):
        vec_GammaH_k = self.Gamma_H[:, :k, k]
        A_k = self.A[:, :k, :k]
        
        xiBWu_k = self.xi_B_Wu[:, :, :k]
        x_0 = np.zeros(self.d_out)
        xiHWu_k = self.solve_xi_H(vec_GammaH_k, xiBWu_k, A_k, x_0)
        self.xi_H_Wu[:, :, k] = xiHWu_k

        xiBWe_k = self.xi_B_We[:, :, :k]
        x_0 = self.sigma_we2 * self.x
        xiHWe_k = self.solve_xi_H(vec_GammaH_k, xiBWe_k, A_k, x_0)
        self.xi_H_We[:, :, k] = xiHWe_k

    def xi_B_equation(self, k):
        vec_GammaB_k = self.Gamma_B[:, :k, k]
        tildeA_k = self.A_tilde[:, :k, :k]

        xiHWu_k = self.xi_H_Wu[:, :, :k]
        x_f = self.sigma_wu2 * self.grad_loss_array[k]
        xiBWu_k = self.solve_xi_H(vec_GammaB_k, xiHWu_k, tildeA_k, x_f, final_condition=True)
        self.xi_B_Wu[:, :, k] = xiBWu_k

        xiHWe_k = self.xi_H_We[:, :, :k]
        x_f = np.zeros(self.d_in)
        xiBWe_k = self.solve_xi_H(vec_GammaB_k, xiHWe_k, tildeA_k, x_f, final_condition=True)
        self.xi_B_We[:, :, k] = xiBWe_k


    def diag_Gamma_equations(self, k):
        # --- 1. Solve for Gamma_H ---
        vecGammaH_k = self.Gamma_H[:, :k, k]        # (num_s, k)
        A_k = self.A[:, :k, :k]                     # (num_s, k, k)
        Lambda_row_for_H = self.Lambda_HB[:, k, :k] # (num_s, k)
        gammaH0 = self.sigma_we2 * self.norm_x2

        gammaH_ts = self._solve_diagonal_ode(
            vecGammaH_k, A_k, Lambda_row_for_H, gammaH0, "Gamma_H"
        )
        
        # --- 2. Solve for Gamma_B ---
        vecGammaB_k = self.Gamma_B[:, :k, k]          # (num_s, k)
        Atilde_k = self.A_tilde[:, :k, :k]            # (num_s, k, k)
        Lambda_col_for_B = self.Lambda_HB[:, :k, k]   # (num_s, k)
        grad_k = self.grad_loss_array[k]
        gammaBf = self.sigma_wu2 * self.inner(grad_k, grad_k)

        gammaB_ts = self._solve_diagonal_ode(
            vecGammaB_k, Atilde_k, Lambda_col_for_B, gammaBf, "Gamma_B", final_condition=True
        )

        # --- 3. Store results ---
        self.Gamma_H[:, k, k] = gammaH_ts
        self.Gamma_B[:, k, k] = gammaB_ts

        # Lambda_{k,k} is identically zero by assumption; ensure it:
        constant = self.xi_B_We[0, :, k].T @ self.x
        self.Lambda_HB[:, k, k] = constant

    def fill_MF_MG(self, k):
        self.M_F1[:, k, k] = 0
        self.M_F2[:, k, k] = 1
        self.M_G1[:, k, k] = 1
        self.M_G2[:, k, k] = 0
        vecGammaH_k = self.Gamma_H[:, :k, k]
        vecGammaB_k = self.Gamma_B[:, :k, k]
        self.M_F1[:, :k, k] = (-self.a * self.eta_v) * (np.einsum('nij,nj->ni', self.M_G1[:, :k, :k], vecGammaB_k))
        self.M_F2[:, :k, k] = (-self.a * self.eta_v) * (np.einsum('nij,nj->ni', self.M_G2[:, :k, :k], vecGammaB_k))
        self.M_G1[:, :k, k] = (-self.a * self.eta_u) * (np.einsum('nij,nj->ni', self.M_F1[:, :k, :k], vecGammaH_k))
        self.M_G2[:, :k, k] = (-self.a * self.eta_u) * (np.einsum('nij,nj->ni', self.M_F2[:, :k, :k], vecGammaH_k))


    def fill_A(self, k):
        """
        Build A_{k+1}(s) from A_k(s), v_Fk(s), v_Gk(s), and a_{k+1}(s).
        """

        GammaHk = self.Gamma_H[:, :k, k]       # (num_s, k)
        GammaBk = self.Gamma_B[:, :k, k]       # (num_s, k)
        GH = self.Gamma_H[:, :k, :k]           # Gamma^H_{∧k}
        GB = self.Gamma_B[:, :k, :k]           # Gamma^B_{∧k}
        MF1 = self.M_F1[:, :k, :k]
        MF2 = self.M_F2[:, :k, :k]
        MG1 = self.M_G1[:, :k, :k]
        MG2 = self.M_G2[:, :k, :k]
        A_k = self.A[:, :k, :k]

        CF = (-self.a**2) * (-self.a * self.eta_u * self.eta_v)

        termF = (
            self.sigma_u2 * (MF1 + self.transpose(MF1)) 
            - self.a*self.eta_u * (
                self.sigma_u2 * np.einsum('nij,njk,nkl->nil', self.transpose(MF1), GH, MF1)
                + self.sigma_v2 * np.einsum('nij,njk,nkl->nil', self.transpose(MF2), GB, MF2)
            )
        )
        vF = CF * np.einsum('nij,nj->ni', termF, GammaHk)

        termG = (
            self.sigma_v2 * (MG2 + self.transpose(MG2)) 
            - self.a*self.eta_v * (
                self.sigma_u2 * np.einsum('nij,njk,nkl->nil', self.transpose(MG1), GH, MG1)
                + self.sigma_v2 * np.einsum('nij,njk,nkl->nil', self.transpose(MG2), GB, MG2)
            )
        )
        vG = CF * np.einsum('nij,nj->ni', termG, GammaBk)

        # a_{k+1}(s)
        cross = np.einsum('ni,nij,nj->n', GammaHk, A_k, GammaBk)
        a_next = (-self.a**2) * (
            (self.eta_u*self.sigma_v2 + self.eta_v*self.sigma_u2)
            - self.eta_u*self.eta_v * cross
        )

        # Now build the block matrix for A_{k+1}
        self.A[:, :k, k] = vF
        self.A[:, k, :k] = vG
        self.A[:, k, k] = a_next

    def fill_W(self, k):
        vecGammaHk = self.Gamma_H[:, :k, k]       # (num_s, k)
        vecGammaBk = self.Gamma_B[:, :k, k]
        A_k = self.A[:, :k, :k]
        tildeA_k = self.A_tilde[:, :k, :k]
        self.W_h[:, k, :k] = np.einsum('nij,nj->ni', self.transpose(A_k), vecGammaHk)
        self.W_b[:, k, :k] = np.einsum('nij,nj->ni', self.transpose(tildeA_k), vecGammaBk)

    def run_single(self, k):
        print(f"Running step {k}/{self.K}")
        # Run Equation for GammaH
        print("Running H_equation")
        self.H_equation(k)
        # Run Equation for xiH
        print("Running xi_H_equation")
        self.xi_H_equation(k)
        # Set values of f_k and the loss gradient at it
        print("Setting f_k and loss gradient")
        self.f[k] = self.xi_H_Wu[-1, :, k]
        self.grad_loss_array[k] = self.grad_loss(self.f[k])
        # Run Equation for GammaB
        print("Running B_equation")
        self.B_equation(k)
        # Run Equation for xiB
        print("Running xi_B_equation")
        self.xi_B_equation(k)
        # Run Equation for diagonal terms in GammaH and GammaB
        print("Running diag_Gamma_equations")
        self.diag_Gamma_equations(k)

        # NOW, Fill all the blanks:
        # Fill MF and MG
        print("Running fill_MF_MG")
        self.fill_MF_MG(k)
        # Fill A
        print("Running fill_A")
        self.fill_A(k)
        # Fill W
        print("Running fill_W")
        self.fill_W(k)


    def run(self):

        for k in range(1, self.K):
            self.run_single(k)

        return {
            'Gamma_H': self.Gamma_H,
            'Gamma_B': self.Gamma_B,
            'Lambda_HB': self.Lambda_HB,
            'A': self.A,
            'W_h': self.W_h,
            'W_b': self.W_b,
            'M_F1': self.M_F1,
            'M_F2': self.M_F2,
            'M_G1': self.M_G1,
            'M_G2': self.M_G2,
            'xi_H_We': self.xi_H_We,
            'xi_H_Wu': self.xi_H_Wu,
            'xi_B_We': self.xi_B_We,
            'xi_B_Wu': self.xi_B_Wu,
            'f': self.f,
            's': self.s
        }


    def solve_xi_H(self, Gamma_s, xi_B_s, A_k_s, x_0, final_condition=False):
        """
        Solve d/ds xi_H(s) = xi_B(s) @ A_k(s).T @ Gamma(s)

        Inputs:
            Gamma_s  : (num_s,)
            xi_B_s   : (num_s, d_out, k)
            A_k_s    : (num_s, k, k)
            x_0      : (d_out,)   # initial (default) or final (if final_condition=True)

        Output:
            xi_H_s   : (num_s, d_out, k)
        """

        num_s, d_out, k = xi_B_s.shape
        assert x_0.shape == (d_out,)

        # Build interpolators
        Gamma_interp = self.interpolate(Gamma_s)
        xiB_interp   = self.interpolate(xi_B_s)
        A_interp     = self.interpolate(A_k_s)

        def rhs(s, xi_flat):
            xiB   = xiB_interp(s)      # (d_out, k)
            A     = A_interp(s)        # (k, k)
            Gamma = Gamma_interp(s)    # (k,)

            # RHS: (d_out,)
            dx = xiB @ A.T @ Gamma

            return dx.reshape(-1)       # flatten

        chosen_method = 'RK45'

        if not final_condition:
            # ----------------------------------------------------
            # Forward solve: x_0 = xi_H(0)
            # ----------------------------------------------------
            sol = solve_ivp(
                rhs,
                (self.s[0], self.s[-1]),
                x_0.reshape(-1),
                t_eval=self.s,
                method=chosen_method,
                rtol=self.rtol,
                atol=self.atol
            )

            if not sol.success:
                raise RuntimeError("Forward ODE solver failed: " + sol.message)

            return sol.y.T.reshape(num_s, d_out)

        else:
            # ----------------------------------------------------
            # Backward solve: x_0 = xi_H(1)
            # ----------------------------------------------------
            sol = solve_ivp(
                rhs,
                (self.s[-1], self.s[0]),
                x_0.reshape(-1),
                t_eval=self.s[::-1],          # decreasing
                method=chosen_method,
                rtol=self.rtol,
                atol=self.atol
            )

            if not sol.success:
                raise RuntimeError("Backward ODE solver failed: " + sol.message)

            # Reverse time order back to increasing s
            xi = sol.y.T[::-1]

            return xi.reshape(num_s, d_out)


    def solve_with_solve_ivp(self, MH, y0, method='RK45', stiff=False, final_condition=False):
        """
        Solve y'(s) = M(s) y(s), where M is given over a grid.
        Supports:
        - forward integration with initial condition y(0) = y0
        - backward integration with final condition y(1) = y0
            (enabled via final_condition=True)

        MH: ndarray shape (num_s, 2k, 2k)
        y0: initial or final condition (2k,)
        final_condition: if True, treat y0 as y(1)
        """

        assert MH.ndim == 3
        num_s, n2, n2b = MH.shape
        assert n2 == n2b
        k2 = n2
        assert y0.shape[0] == k2

        M_interp = self.interpolate(MH)

        def rhs(s, y):
            return M_interp(s).dot(y)

        chosen_method = 'BDF' if stiff else method

        if not final_condition:
            # -----------------------
            # Standard forward solve
            # -----------------------
            sol = solve_ivp(
                rhs,
                (self.s[0], self.s[-1]),
                y0,
                method=chosen_method,
                t_eval=self.s,
                rtol=self.rtol,
                atol=self.atol
            )
            if not sol.success:
                raise RuntimeError("ODE solver failed: " + str(sol.message))
            return sol.y.T

        else:
            # --------------------------------------
            # Backward solve: treat y0 = y(1)
            # Integrate from s=1 down to s=0
            # --------------------------------------
            sol = solve_ivp(
                rhs,
                (self.s[-1], self.s[0]),
                y0,
                method=chosen_method,
                t_eval=self.s[::-1],      # decreasing order
                rtol=self.rtol,
                atol=self.atol
            )
            if not sol.success:
                raise RuntimeError("Backward ODE solve failed: " + str(sol.message))

            # sol.y has shape (2k, num_s decreasing)
            # Reverse to match increasing self.s
            return sol.y.T[::-1]


    def _solve_diagonal_ode(self, vec_gamma_data, A_data, lambda_data, y0, name, final_condition=False):
        """
        Helper method to solve a single scalar diagonal ODE.

        Args:
            vec_gamma_data (np.ndarray): Time-series data for the Gamma vector (shape (num_s, k)).
            A_data (np.ndarray): Time-series data for the A matrix (shape (num_s, k, k)).
            lambda_data (np.ndarray): Time-series data for the Lambda vector (shape (num_s, k)).
            y0 (float): Initial condition y(s0) or final condition y(s_f).
            name (str): Name for error logging (e.g., "Gamma_H").
            final_condition (bool): If True, treat y0 as y(s[-1]) and solve backwards.

        Returns:
            np.ndarray: The solution time-series (shape (num_s,)).
        """

        # --- build interpolators ---
        interp_vecG = self.interpolate(vec_gamma_data)  # returns (k,)
        interp_A = self.interpolate(A_data)  # returns (k,k)
        interp_L = self.interpolate(lambda_data)  # returns (k,)

        # --- RHS function (scalar) ---
        def rhs(s, y):
            # y is scalar (unused on RHS except to satisfy signature)
            vecG = interp_vecG(s)  # (k,)
            A = interp_A(s)  # (k,k)
            L = interp_L(s)  # (k,)
            
            # Compute 2.0 * (vecG @ A @ L)
            tmp = A.dot(L)  # (k,)
            value = 2.0 * float(np.dot(vecG, tmp))
            return value

        # --- integrate scalar ODE ---
        
        if not final_condition:
            # Standard forward solve: y(s[0]) = y0
            t_span = (self.s[0], self.s[-1])
            t_eval = self.s
        else:
            # Backward solve: y(s[-1]) = y0
            t_span = (self.s[-1], self.s[0]) # From s=1 to s=0
            t_eval = self.s[::-1]            # Decreasing order
        
        sol = solve_ivp(rhs, t_span, [y0],
                        t_eval=t_eval, method='RK45', rtol=self.rtol, atol=self.atol)
        
        if not sol.success:
            raise RuntimeError(f"{name} diagonal solve failed: " + sol.message)

        if final_condition:
            # Reverse the time series back to increasing self.s
            return sol.y.ravel()[::-1]
        else:
            return sol.y.ravel() # shape (num_s,)
