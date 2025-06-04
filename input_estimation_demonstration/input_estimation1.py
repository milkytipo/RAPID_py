import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd # Import pandas for the DataFrame table

# --- System Model Definitions (as provided previously) ---
system1 = {
    "name": "Original System",
    "A": np.array([[1.0, 1.0], [0.0, 1.0]]),
    "G": np.array([[0.5], [1.0]]),
    "C": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "Q_factor": 0.01, "R_factor": 0.1,
    "n": 2, "m": 1, "p": 2
}
system2 = {
    "name": "Input Affects First State, Both Observed",
    "A": np.array([[0.9, 0.2], [-0.1, 0.8]]), "G": np.array([[1.0], [0.0]]),
    "C": np.array([[1.0, 0.0], [0.0, 1.0]]), "Q_factor": 0.02, "R_factor": 0.15,
    "n": 2, "m": 1, "p": 2
}
system3 = {
    "name": "Input to x2, Mixed Observations",
    "A": np.array([[0.8, 0.0], [0.3, 0.7]]), "G": np.array([[0.0], [1.0]]),
    "C": np.array([[1.0, 0.0], [1.0, 1.0]]), "Q_factor": 0.015, "R_factor": 0.05,
    "n": 2, "m": 1, "p": 2
}
system4 = {
    "name": "Oscillating System",
    "A": np.array([[0.9 * np.cos(np.pi/6), -np.sin(np.pi/6)], [np.sin(np.pi/6), 0.9 * np.cos(np.pi/6)]]),
    "G": np.array([[0.5], [0.5]]), "C": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "Q_factor": 0.01, "R_factor": 0.1,
    "n": 2, "m": 1, "p": 2
}
system5 = {
    "name": "3 States, 1 Input, 2 Outputs",
    "A": np.array([[0.9, 0.1, 0.05], [0.0, 0.8, 0.15], [0.1, -0.1, 0.75]]),
    "G": np.array([[1.0], [0.0], [0.5]]), "C": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "Q_factor": 0.005, "R_factor": 0.08,
    "n": 3, "m": 1, "p": 2
}

systems_to_test = [system1, system2, system3, system4, system5]
seed_value = 42
np.random.seed(seed_value)

# --- Main loop to iterate through systems ---
for system_params in systems_to_test:
    print(f"\n--- Testing System: {system_params['name']} ---")

    n = system_params["n"]; m = system_params["m"]; p = system_params["p"]
    A = system_params["A"]; G = system_params["G"]; C = system_params["C"]
    Q = system_params["Q_factor"] * np.eye(n); R = system_params["R_factor"] * np.eye(p)

    F_check = C @ G
    rank_F = np.linalg.matrix_rank(F_check)
    if rank_F != m:
        print(f"WARNING: System '{system_params['name']}' does NOT satisfy rank condition! rank(CG)={rank_F}, m={m}. Skipping...")
        continue
    else:
        print(f"System '{system_params['name']}' satisfies rank condition: rank(CG)={rank_F}, m={m}")

    N = 200 # Number of time steps

    # Initialize states and covariances for all filters
    x_prev_umv = np.zeros((n, 1)); P_prev_umv = np.eye(n) * 0.1
    x_prev_kf = np.zeros((n, 1)); P_prev_kf = np.eye(n) * 0.1
    x_prev_umv_L = np.zeros((n, 1)); P_prev_umv_L = np.eye(n) * 0.1
    x_prev_naive_umv = np.zeros((n, 1)); P_prev_naive_umv = np.eye(n) * 0.1
    
    x_true_current_step = np.random.multivariate_normal(np.zeros(n), P_prev_umv).reshape(-1,1) # Initial true state

    # Histories
    x_true_hist = []; x_est_umv_hist = []; x_est_kf_hist = []
    x_est_umv_L_hist = []; x_est_naive_umv_hist = []
    d_est_umv_hist = []; d_est_umv_L_hist = []; d_est_naive_umv_hist = []

    # Simulation loop
    for step in range(N):
        d_true_val = 0 # Constant true input
        d_true = d_true_val * np.ones((m, 1))
        w = np.random.multivariate_normal(np.zeros(n), Q).reshape(-1, 1)
        v = np.random.multivariate_normal(np.zeros(p), R).reshape(-1, 1)
        
        if step > 0 : # Use last true state for propagation
             x_true_current_step = A @ x_true_hist[-1].reshape(-1,1) + G @ d_true + w
        # else: use the initialized x_true_current_step

        y = C @ x_true_current_step + v

        # --- UMV Filter (Gillijns and De Moor - Optimal K) ---
        x_pred_umv = A @ x_prev_umv
        P_pred_umv = A @ P_prev_umv @ A.T + Q
        y_tilde_umv = y - C @ x_pred_umv
        F_umv = C @ G
        R_tilde_umv = C @ P_pred_umv @ C.T + R
        M_umv = np.zeros((m,p)) # Default in case of error
        d_est_umv = np.zeros((m,1))
        try:
            inv_R_tilde_umv = np.linalg.inv(R_tilde_umv)
            M_inv_term_umv = F_umv.T @ inv_R_tilde_umv @ F_umv
            M_umv = np.linalg.inv(M_inv_term_umv) @ F_umv.T @ inv_R_tilde_umv
            d_est_umv = M_umv @ y_tilde_umv
        except np.linalg.LinAlgError: pass # d_est_umv remains zero
        x_sharp_umv = x_pred_umv + G @ d_est_umv
        P_sharp_umv = (np.eye(n) - G @ M_umv @ C) @ P_pred_umv @ (np.eye(n) - G @ M_umv @ C).T + G @ M_umv @ R @ M_umv.T @ G.T
        S_sharp_umv = -G @ M_umv @ R
        R_sharp_umv = (np.eye(p) - C @ G @ M_umv) @ R_tilde_umv @ (np.eye(p) - C @ G @ M_umv).T
        U_rsh_umv, s_rsh_umv, _ = la.svd(R_sharp_umv)
        K_opt_umv = np.zeros((n,p)) # Default K
        if s_rsh_umv.size > 0: threshold_rsh_umv = np.max(s_rsh_umv) * 1e-7
        else: threshold_rsh_umv = 1e-7
        rank_r_rsh_umv = np.sum(s_rsh_umv > threshold_rsh_umv)
        if rank_r_rsh_umv > 0:
            alpha_umv = U_rsh_umv[:, :rank_r_rsh_umv].T
            num_umv = (P_sharp_umv @ C.T + S_sharp_umv) @ alpha_umv.T
            den_umv = alpha_umv @ R_sharp_umv @ alpha_umv.T
            try: K_opt_umv = num_umv @ np.linalg.inv(den_umv) @ alpha_umv
            except np.linalg.LinAlgError: pass # K_opt_umv remains zero
        x_updated_umv = x_sharp_umv + K_opt_umv @ (y - C @ x_sharp_umv)
        P_updated_umv = P_sharp_umv - K_opt_umv @ (P_sharp_umv @ C.T + S_sharp_umv).T

        # --- Classic Kalman Filter ---
        x_pred_kf = A @ x_prev_kf
        P_pred_kf = A @ P_prev_kf @ A.T + Q
        y_tilde_kf = y - C @ x_pred_kf
        S_kf = C @ P_pred_kf @ C.T + R
        K_kf = np.zeros((n,p)) # Default
        try: K_kf = P_pred_kf @ C.T @ np.linalg.inv(S_kf)
        except np.linalg.LinAlgError: pass
        x_updated_kf = x_pred_kf + K_kf @ y_tilde_kf
        P_updated_kf = (np.eye(n) - K_kf @ C) @ P_pred_kf
        
        # --- UMV Filter (Correct L - from your original code structure) ---
        x_pred_umv_L = A @ x_prev_umv_L
        P_pred_umv_L = A @ P_prev_umv_L @ A.T + Q
        CG_L = C @ G
        B_L = np.linalg.pinv(CG_L) # Using pseudo-inverse for B
        d_est_umv_L = B_L @ (y - C @ x_pred_umv_L)
        residual_L = y - C @ x_pred_umv_L - CG_L @ d_est_umv_L
        I_n_L = np.eye(n)
        term_L_inv = C @ (I_n_L - G @ B_L @ C) @ P_pred_umv_L @ C.T + R # This structure for L's innovation cov is one form.
                                                                       # The paper for Kitanidis/Darouach gain L_k = P_k|k-1 H_k^T (H_k P_k|k-1 H_k^T + M_k R M_k^T)^{-1}
                                                                       # where H_k = C_k (I-G M_k C_k), M_k = (C_k G_k-1 M_k C_k G)^{-1} C_k G ...
                                                                       # Your original L: L = (I - G @ B) @ P_pred_umv_L @ C.T @ np.linalg.inv(C @ (I - G @ B) @ P_pred_umv_L @ C.T + R)
        L_gain = np.zeros((n,p)) # Default
        try:
            # Using your formulation for L's denominator from your original code
            # Note: The term (I - G @ B @ C) is used in some derivations for projecting out input.
            # If B_L = M_k from Gillijns, then (I - G M C) is used in P_sharp.
            # Let's use the structure you had for L for "CorrectL" as per your code.
            # The P_pred_umv_L for the C term should be (I - G @ B_L @ C) @ P_pred_umv_L
            # Let projector_L = (I_n_L - G @ B_L @ C)
            projector_L = (I_n_L - G @ B_L @ C) # More standard form (I - G M C)
            # If B_L is seen as the input gain part M
            # Then L is related to the Kitanidis gain structure where state is corrected first.
            # The term inside inv is C @ P_corr @ C.T + R where P_corr is P after input effect removed.
            # P_corr_L = projector_L @ P_pred_umv_L @ projector_L.T + G @ B_L @ R @ B_L.T @ G.T
            # innovation_cov_L = C @ P_corr_L @ C.T + R
            # K_equiv_L = P_corr_L @ C.T @ np.linalg.inv(innovation_cov_L)
            # x_umv_L_temp = x_pred_umv_L + G @ d_est_umv_L
            # x_updated_umv_L = x_umv_L_temp + K_equiv_L @ (y - C @ x_umv_L_temp)
            
            # Sticking to your original "CorrectL" structure:
            L_term_P = (I_n_L - G @ B_L @ C) @ P_pred_umv_L # Using your (I - G @ B) form might imply B=M_k (C_k G_{k-1})^{-1}
            # For simplicity and to match what you called "Correct L", let's try to replicate it.
            # The structure you had: L = (I - G @ B) @ P_pred_umv_L @ C.T @ np.linalg.inv(C @ (I - G @ B) @ P_pred_umv_L @ C.T + R)
            # Here G@B is (n x m) @ (m x p) = (n x p). (I - G@B) is (n x n) if G@B is nxn, which means p=n.
            # This is only possible if p=n. If G@B@C, then it's (n x n).
            # Let's assume B_L is the input gain M_k: M_k = (F.T R_tilde_inv F)^-1 F.T R_tilde_inv
            # For this filter, let's use the Kitanidis-style gain Lk as defined by Gillijns & De Moor, Eq. (19) and (20).
            # L_k = K_k + (I_n - K_k C_k) G_{k-1} M_k
            # where K_k = P_{k|k-1} C_k^T \tilde{R}_k^{-1} for Kitanidis state update part (Theorem 5)
            # and M_k is the optimal input gain.
            # This leads to state update: \hat{x}_{k|k} = \hat{x}_{k|k-1} + L_k (y_k - C_k \hat{x}_{k|k-1})
            # This is essentially what the first UMV filter does if K_opt is used as L_k.
            # Your UMV Filter (Correct L) seemed to aim for a different formulation.
            # Let's re-evaluate what "Correct L" meant in your original context.
            # If B = pinv(CG), it's a direct least-squares for input given innovation.
            # x_umv_L = x_pred_umv_L + G @ d_est_umv_L + L @ residual_L
            # This is a three-stage update. This is similar to Friedland's two-stage filter.
            # The "L" gain you used seems to be a standard Kalman gain on a transformed system.
            # Let's implement the three-stage filter as you had it, using K_like for L.
            P_intermediate_L = (I_n_L - G @ B_L @ C) @ P_pred_umv_L # Covariance after input estimation (approx)
            S_L_denom = C @ P_intermediate_L @ C.T + R # Innovation cov for the residual update
            L_gain = P_intermediate_L @ C.T @ np.linalg.inv(S_L_denom)

        except np.linalg.LinAlgError: pass # L_gain remains zero
        x_updated_umv_L = x_pred_umv_L + G @ d_est_umv_L + L_gain @ residual_L # Three-stage update
        P_updated_umv_L = (I_n_L - L_gain @ C) @ P_intermediate_L # Approx covariance


        # --- UMV Filter (Naive K - from your original code structure) ---
        x_pred_naive_umv = A @ x_prev_naive_umv
        P_pred_naive_umv = A @ P_prev_naive_umv @ A.T + Q
        CG_naive = C @ G
        B_naive = np.linalg.pinv(CG_naive) # Using pseudo-inverse
        d_est_naive_umv = B_naive @ (y - C @ x_pred_naive_umv)
        residual_naive_umv = y - C @ x_pred_naive_umv - CG_naive @ d_est_naive_umv # Residual after input compensation
        
        K_naive_gain = np.zeros((n,p)) # Default
        try:
            S_naive_denom = C @ P_pred_naive_umv @ C.T + R # Standard KF innovation cov (ignores input effect on this cov)
            K_naive_gain = P_pred_naive_umv @ C.T @ np.linalg.inv(S_naive_denom)
        except np.linalg.LinAlgError: pass # K_naive_gain remains zero
            
        x_updated_naive_umv = x_pred_naive_umv + G @ d_est_naive_umv + K_naive_gain @ residual_naive_umv # Three-stage update
        # Covariance for Naive K is tricky. The K_naive is based on P_pred_naive which doesn't account for input.
        # (I - K C)P is for standard KF.
        # For a three-stage, it's more complex. Let's use a simplified one from your original:
        P_updated_naive_umv = (np.eye(n) - K_naive_gain @ C) @ P_pred_naive_umv


        # Save results
        x_true_hist.append(x_true_current_step.flatten())
        x_est_umv_hist.append(x_updated_umv.flatten())
        x_est_kf_hist.append(x_updated_kf.flatten())
        x_est_umv_L_hist.append(x_updated_umv_L.flatten())
        x_est_naive_umv_hist.append(x_updated_naive_umv.flatten())
        
        d_est_umv_hist.append(d_est_umv.flatten())
        if m == 1: # Assuming scalar input for easy history append
            d_est_umv_L_hist.append(d_est_umv_L.item())
            d_est_naive_umv_hist.append(d_est_naive_umv.item())
        else: # For vector input, append as list or handle appropriately
            d_est_umv_L_hist.append(d_est_umv_L.flatten().tolist())
            d_est_naive_umv_hist.append(d_est_naive_umv.flatten().tolist())


        # Update for next step
        x_prev_umv = x_updated_umv; P_prev_umv = P_updated_umv
        x_prev_kf = x_updated_kf; P_prev_kf = P_updated_kf
        x_prev_umv_L = x_updated_umv_L; P_prev_umv_L = P_updated_umv_L
        x_prev_naive_umv = x_updated_naive_umv; P_prev_naive_umv = P_updated_naive_umv
        
        # Update true state for the next simulation step if it's not the first step
        if step == 0 and N > 1 : # if it was the first step and there are more steps
            x_true_hist[-1] = x_true_current_step.flatten() # ensure the first true state is stored correctly
        elif step > 0:
            x_true_hist[-1] = x_true_current_step.flatten()



    # Convert histories to NumPy arrays
    x_true_hist_np = np.array(x_true_hist)
    x_est_umv_hist_np = np.array(x_est_umv_hist)
    x_est_kf_hist_np = np.array(x_est_kf_hist)
    x_est_umv_L_hist_np = np.array(x_est_umv_L_hist)
    x_est_naive_umv_hist_np = np.array(x_est_naive_umv_hist)
    
    d_est_umv_hist_np = np.array(d_est_umv_hist)
    d_est_umv_L_hist_np = np.array(d_est_umv_L_hist)
    d_est_naive_umv_hist_np = np.array(d_est_naive_umv_hist)

    # --- RMSE Calculation & Table ---
    rmse_results_list = []
    method_names = ["UMV (Optimal K)", "KF (No Input Est)", "UMV (Correct L-like)", "UMV (Naive K)"]
    state_estimates = [x_est_umv_hist_np, x_est_kf_hist_np, x_est_umv_L_hist_np, x_est_naive_umv_hist_np]

    for i_method, method_name in enumerate(method_names):
        current_rmse_list = [method_name]
        for i_state in range(n):
            rmse_val = np.sqrt(np.mean((x_true_hist_np[:, i_state] - state_estimates[i_method][:, i_state])**2))
            current_rmse_list.append(f"{rmse_val:.4f}")
        rmse_results_list.append(current_rmse_list)

    col_headers = ["Method"] + [f"RMSE x[{i}]" for i in range(n)]
    df_rmse = pd.DataFrame(rmse_results_list, columns=col_headers)

    print("\nRMSE Results:")
    print(df_rmse)

    fig_table, ax_table = plt.subplots(figsize=(max(6, n * 2.5), 2)) # Adjust table size
    ax_table.axis('tight'); ax_table.axis('off')
    table_obj = ax_table.table(cellText=df_rmse.values, colLabels=df_rmse.columns, loc='center', cellLoc='center')
    table_obj.auto_set_font_size(False); table_obj.set_fontsize(10); table_obj.scale(1.1, 1.1)
    plt.title(f"Estimation RMSE Comparison: {system_params['name']}", fontweight='bold', y=0.8) # Adjust title position
    plt.show()


    # --- Plotting ---
    fig, axs = plt.subplots(n + 1, 1, figsize=(12, 3 * (n + 1) + 1), sharex=True)
    fig.suptitle(f"Estimation Results for: {system_params['name']}", fontweight='bold', fontsize=14)

    for i in range(n):
        ax_state = axs[i]
        ax_state.plot(x_true_hist_np[:, i], label=f'True x[{i}]', color='black', linewidth=1.5)
        ax_state.plot(x_est_umv_hist_np[:, i], '--', label=f'UMV (Optimal K) x[{i}]')
        ax_state.plot(x_est_kf_hist_np[:, i], '-.', label=f'KF x[{i}]')
        ax_state.plot(x_est_umv_L_hist_np[:, i], ':', label=f'UMV (Correct L-like) x[{i}]')
        ax_state.plot(x_est_naive_umv_hist_np[:, i], '--', label=f'UMV (Naive K) x[{i}]')
        ax_state.set_ylabel(f"State x[{i}]")
        ax_state.legend(fontsize=8)
        ax_state.grid(True, linestyle=':', alpha=0.7)

    ax_input = axs[n]
    ax_input.plot(d_est_umv_hist_np, '--', label='UMV (Optimal K) Estimated d')
    ax_input.plot(d_est_umv_L_hist_np, ':', label='UMV (Correct L-like) Estimated d')
    ax_input.plot(d_est_naive_umv_hist_np, '--', label='UMV (Naive K) Estimated d')
    ax_input.axhline(d_true_val, color='r', linestyle='--', label=f'True d = {d_true_val}')
    ax_input.set_ylabel("Estimated Input d")
    ax_input.set_xlabel("Time Step")
    ax_input.legend(fontsize=8)
    ax_input.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()