import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def build_model(dt):
    x = ca.SX.sym("x", 2)  # [X, S]
    u = ca.SX.sym("u", 1)  # [F]

    mu_max = 0.42
    ks = 0.12
    yxs = 0.52
    sin = 20.0
    v = 1.0

    mu = mu_max * x[1] / (ks + x[1] + 1e-9)
    d = u[0] / v
    dx = ca.vertcat(
        (mu - d) * x[0],
        d * (sin - x[1]) - (mu / yxs) * x[0],
    )
    x_next = x + dt * dx
    z = ca.vertcat(x[0])  # Only biomass is measured

    f = ca.Function("f", [x, u], [x_next])
    h = ca.Function("h", [x], [z])
    f_jac = ca.Function("f_jac", [x, u], [ca.jacobian(x_next, x)])
    h_jac = ca.Function("h_jac", [x], [ca.jacobian(z, x)])
    return f, h, f_jac, h_jac


def solve_nmpc(x0_est, u_prev, x_sp, f_model, dt, n_horizon=12):
    q = 10.0
    r_du = 0.5
    qf = 50.0
    terminal_band = 0.5
    umin, umax = 0.0, 1.0
    dumax = 0.1

    opti = ca.Opti()
    u = opti.variable(1, n_horizon)
    x = opti.variable(2, n_horizon + 1)

    opti.subject_to(x[:, 0] == x0_est)
    j = 0
    for k in range(n_horizon):
        x_next = f_model(x[:, k], u[:, k])
        opti.subject_to(x[:, k + 1] == x_next)
        opti.subject_to(opti.bounded(umin, u[:, k], umax))
        du = u[:, k] - (u_prev if k == 0 else u[:, k - 1])
        opti.subject_to(opti.bounded(-dumax, du, dumax))
        j += q * (x[0, k + 1] - x_sp) ** 2 + r_du * du**2

    j += qf * (x[0, -1] - x_sp) ** 2
    opti.subject_to(opti.bounded(-terminal_band, x[0, -1] - x_sp, terminal_band))
    opti.minimize(j)

    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        return float(sol.value(u[0, 0])), True, opti.stats().get("return_status", "ok")
    except RuntimeError:
        status = opti.stats().get("return_status", "failed")
        return float(u_prev), False, status


def main():
    np.random.seed(42)
    dt = 0.1
    t_final = 15.0
    n_steps = int(t_final / dt)
    time = np.linspace(0, t_final, n_steps + 1)

    f_model, h_model, f_jac, h_jac = build_model(dt)

    x_real = np.array([[0.6], [16.0]])
    x_est = np.array([[0.4], [18.0]])
    p_est = np.diag([0.2, 0.5])
    q_ekf = np.diag([1e-4, 1e-4])
    r_ekf = np.array([[0.02**2]])

    u_prev = 0.2
    x_setpoint = 1.2

    xr_hist = np.zeros((2, n_steps + 1))
    xe_hist = np.zeros((2, n_steps + 1))
    y_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros(n_steps)
    status_hist = []

    xr_hist[:, 0] = x_real.flatten()
    xe_hist[:, 0] = x_est.flatten()
    y_hist[0] = float(h_model(x_real).full()[0, 0])

    for k in range(n_steps):
        yk = float(h_model(x_real).full()[0, 0] + np.random.normal(0.0, np.sqrt(r_ekf[0, 0])))

        x_pred = f_model(x_est, np.array([u_prev])).full()
        fk = f_jac(x_est, np.array([u_prev])).full()
        p_pred = fk @ p_est @ fk.T + q_ekf

        hk = h_jac(x_pred).full()
        y_pred = h_model(x_pred).full()
        sk = hk @ p_pred @ hk.T + r_ekf
        kk = p_pred @ hk.T @ np.linalg.pinv(sk)
        innov = np.array([[yk]]) - y_pred
        x_est = x_pred + kk @ innov
        x_est = np.maximum(x_est, 1e-8)
        p_est = (np.eye(2) - kk @ hk) @ p_pred

        u_cmd, ok, status = solve_nmpc(x_est.flatten(), u_prev, x_setpoint, f_model, dt)
        if not ok:
            print(f"NMPC warning in step {k}: {status}")
        u_prev = float(np.clip(u_cmd, 0.0, 1.0))

        x_real = f_model(x_real, np.array([u_prev])).full()
        x_real = np.maximum(x_real, 1e-8)

        xr_hist[:, k + 1] = x_real.flatten()
        xe_hist[:, k + 1] = x_est.flatten()
        y_hist[k + 1] = yk
        u_hist[k] = u_prev
        status_hist.append(status)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(time, xr_hist[0], label="X real")
    axs[0].plot(time, xe_hist[0], "--", label="X estimada (EKF)")
    axs[0].axhline(x_setpoint, color="r", linestyle=":", label="SP X")
    axs[0].set_ylabel("Biomasa (g/L)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time, xr_hist[1], label="S real")
    axs[1].plot(time, xe_hist[1], "--", label="S estimada (EKF)")
    axs[1].set_ylabel("Sustrato (g/L)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].step(time[:-1], u_hist, where="post", label="F (NMPC)")
    axs[2].set_xlabel("Tiempo (h)")
    axs[2].set_ylabel("Flujo de alimentación (L/h)")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
