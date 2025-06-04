import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

seed_value = 42  # Or any integer you choose
np.random.seed(seed_value)
# Set system dimensions
n = 2  # state dimension
m = 1  # unknown input dimension
p = 2  # output dimension, with p > m

# System matrices
A = np.array([[1.0, 1.0], [0.0, 1.0]])
G = np.array([[0.5], [1.0]])
C = np.array([[1.0, 0.0], [0.0, 1.0]])

Q = 0.01 * np.eye(n)
R = 0.1 * np.eye(p)

# Number of time steps
N = 500

# Initialize state and covariance
x_prev_umv = np.zeros((n, 1))
P_prev_umv = np.eye(n)

x_prev_kf = np.zeros((n, 1))
P_prev_kf = np.eye(n)

x_umv = np.zeros((n, 1))
P_umv = np.eye(n)

x_umv_L = np.zeros((n, 1))
P_umv_L = np.eye(n)

x_naive_umv = np.zeros((n, 1))
P_naive_umv = np.eye(n)

# Histories
x_true_hist = []
x_est_umv_hist = []
x_est_kf_hist = []
d_est_hist = []
x_umv_hist = []
u_umv_hist = []
x_naive_umv_hist = []
u_naive_umv_hist = []


# Simulation loop
for _ in range(N):
    # Simulate true system with unknown input = 0
    d_true = 5 * np.ones((m, 1))
    w = np.random.multivariate_normal(np.zeros(n), Q).reshape(-1, 1)
    v = np.random.multivariate_normal(np.zeros(p), R).reshape(-1, 1)
    x_true = A @ x_prev_umv + G @ d_true + w
    y = C @ x_true + v

    ### UMV FILTER ###
    x_pred_umv = A @ x_prev_umv
    P_pred_umv = A @ P_prev_umv @ A.T + Q

    y_tilde = y - C @ x_pred_umv
    F = C @ G
    R_tilde = C @ P_pred_umv @ C.T + R
    M = np.linalg.inv(F.T @ np.linalg.inv(R_tilde) @ F) @ F.T @ np.linalg.inv(R_tilde)
    d_est = M @ y_tilde
    x_sharp = x_pred_umv + G @ d_est

    P_sharp = (np.eye(n) - G @ M @ C) @ P_pred_umv @ (np.eye(n) - G @ M @ C).T + G @ M @ R @ M.T @ G.T
    S_sharp = -G @ M @ R
    R_sharp = (np.eye(p) - C @ G @ M) @ R_tilde @ (np.eye(p) - C @ G @ M).T
    U, _, _ = la.svd(R_sharp)
    alpha = U[:, :p - m].T
    numerator = (P_sharp @ C.T + S_sharp) @ alpha.T
    denominator = alpha @ R_sharp @ alpha.T
    K_opt = numerator @ np.linalg.inv(denominator) @ alpha
    residual = y - C @ x_sharp
    x_updated_umv = x_sharp + K_opt @ residual


    ### CLASSIC KALMAN FILTER (ignores unknown input) ###
    x_pred_kf = A @ x_prev_kf
    P_pred_kf = A @ P_prev_kf @ A.T + Q
    S_kf = C @ P_pred_kf @ C.T + R
    K_kf = P_pred_kf @ C.T @ np.linalg.inv(S_kf)
    x_updated_kf = x_pred_kf + K_kf @ (y - C @ x_pred_kf)
    P_updated_kf = (np.eye(n) - K_kf @ C) @ P_pred_kf


    ### UMV Filter (Correct L) ###
    # x_pred_umv_L = A @ x_umv_L
    # P_pred_umv_L = A @ P_umv_L @ A.T + Q
    # CG = C @ G
    # B = np.linalg.pinv(CG)

    # F = C @ G
    # R_tilde = C @ P_pred_umv_L @ C.T + R
    # M = np.linalg.inv(F.T @ np.linalg.inv(R_tilde) @ F) @ F.T @ np.linalg.inv(R_tilde)
    # u_est = M @ (y - C @ x_pred_umv_L)
    # residual = y - C @ x_pred_umv_L - CG @ u_est
    # I = np.eye(n)
    # # L = (I - B @ G) @ P_pred_umv_L @ C.T @ np.linalg.inv(C @ (I - B @ G) @ P_pred_umv_L @ C.T + R)
    # L = (I - G @ B) @ P_pred_umv_L @ C.T @ np.linalg.inv(C @ (I - G @ B) @ P_pred_umv_L @ C.T + R)
    # x_umv_L = x_pred_umv_L + G @ u_est + L @ residual
    # P_umv_L = (I - L @ C) @ P_pred_umv_L

    ### UMV Filter (Naive K) ###
    x_pred_naive = A @ x_naive_umv
    P_pred_naive = A @ P_naive_umv @ A.T + Q
    F = C @ G
    R_tilde = C @ P_pred_naive @ C.T + R
    M = np.linalg.inv(F.T @ np.linalg.inv(R_tilde) @ F) @ F.T @ np.linalg.inv(R_tilde)
    u_naive = M @ (y - C @ x_pred_naive)
    residual_naive = y - C @ x_pred_naive - G @ u_naive
    K_naive = P_pred_naive @ C.T @ np.linalg.inv(C @ P_pred_naive @ C.T + R)
    x_naive_umv = x_pred_naive + G @ u_naive + K_naive @ residual_naive



    print("diff between UMV and UMV-K:", np.linalg.norm(x_updated_umv - x_naive_umv))
    print("2: diff between UMV and UMV-K:",( K_naive - K_opt) @ residual)
    P_naive_umv = (np.eye(n) - K_naive @ C) @ P_pred_naive

    # Save results
    x_true_hist.append(x_true.flatten())
    x_est_umv_hist.append(x_updated_umv.flatten())
    d_est_hist.append(d_est.flatten())
    x_est_kf_hist.append(x_updated_kf.flatten())

    x_umv_hist.append(x_umv_L.flatten())
    # u_umv_hist.append(u_est.item())
    x_naive_umv_hist.append(x_naive_umv.flatten())
    u_naive_umv_hist.append(u_naive.item())

    # Update for next step
    x_prev_umv = x_updated_umv
    # P_prev_umv = P_sharp
    P_updated_umv = P_sharp - K_opt @ (P_sharp @ C.T + S_sharp).T
    P_prev_umv = P_updated_umv 


    x_prev_kf = x_updated_kf
    P_prev_kf = P_updated_kf

# Convert to arrays for plotting
x_true_hist = np.array(x_true_hist)
x_est_umv_hist = np.array(x_est_umv_hist)
x_est_kf_hist = np.array(x_est_kf_hist)
d_est_hist = np.array(d_est_hist)
x_umv_hist = np.array(x_umv_hist)
u_umv_hist = np.array(u_umv_hist)
x_naive_umv_hist = np.array(x_naive_umv_hist)
u_naive_umv_hist = np.array(u_naive_umv_hist)





# Compute RMSE for each method
rmse = lambda est: np.sqrt(np.mean((x_true_hist - est)**2, axis=0)).tolist()

results = {
    "Method": ["UMV (Optimal)", "KF", "UMV-K"],
    "RMSE x[0]": [
        np.sqrt(np.mean((x_true_hist[:, 0] - x_est_umv_hist[:, 0])**2)),
        np.sqrt(np.mean((x_true_hist[:, 0] - x_est_kf_hist[:, 0])**2)),
        # np.sqrt(np.mean((x_true_hist[:, 0] - x_umv_hist[:, 0])**2)),
        np.sqrt(np.mean((x_true_hist[:, 0] - x_naive_umv_hist[:, 0])**2)),
    ],
    "RMSE x[1]": [
        np.sqrt(np.mean((x_true_hist[:, 1] - x_est_umv_hist[:, 1])**2)),
        np.sqrt(np.mean((x_true_hist[:, 1] - x_est_kf_hist[:, 1])**2)),
        # np.sqrt(np.mean((x_true_hist[:, 1] - x_umv_hist[:, 1])**2)),
        np.sqrt(np.mean((x_true_hist[:, 1] - x_naive_umv_hist[:, 1])**2)),
    ]
}
import pandas as pd
df_rmse = pd.DataFrame(results)
# Display the comparison table using matplotlib
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('tight')
ax.axis('off')
table_data = [["Method", "RMSE x[0]", "RMSE x[1]"]] + df_rmse.values.tolist()
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLabels=None)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("Estimation RMSE Comparison", fontweight='bold')
# plt.show()



# Plot state estimation comparison
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

axs[0].plot(x_true_hist[:, 0], label='True x[0]')
axs[0].plot(x_est_umv_hist[:, 0], '--', label='UMV x[0]')
axs[0].plot(x_est_kf_hist[:, 0], '-.', label='KF x[0]')
axs[0].plot(x_naive_umv_hist[:, 0], '-.', label='UMV-K x[0]')
# axs[0].plot(x_umv_hist[:, 0], '-.',label='UMV-L x[0]')
axs[0].set_title("State x[0]")
axs[0].legend()

axs[1].plot(x_true_hist[:, 1], label='True x[1]')
axs[1].plot(x_est_umv_hist[:, 1], '--', label='UMV x[1]')
axs[1].plot(x_est_kf_hist[:, 1], '-.', label='KF x[1]')
axs[1].plot(x_naive_umv_hist[:, 1], '-.', label='UMV-K x[1]')
# axs[1].plot(x_umv_hist[:, 1],  label='UMV-L x[1]')

axs[1].set_title("State x[1]")
axs[1].legend()

axs[2].plot(d_est_hist, label='UMV Estimated d')
axs[2].plot(u_naive_umv_hist, '--', label='UMV-K Estimated u (K)')
# axs[2].plot(u_umv_hist,  label='UMV-L Estimated u (K)')
axs[2].axhline(0, color='r', linestyle='--', label='True d = 0')

axs[2].set_title("Estimated Unknown Input d")
axs[2].legend()

plt.tight_layout()
plt.show()
