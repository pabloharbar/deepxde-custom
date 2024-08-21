import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("tableau-colorblind10")


def plot_training_points(x, y, simulation_name: str):
    plt.figure(figsize=(20, 4))
    plt.scatter(x, y, s=0.5)
    plt.title("Training points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"./{simulation_name}/training_points.png")
    # plt.show()
    plt.close()


def plot_loss_history(loss_test, loss_steps, simulation_name: str, plot_name: str):
    plot_array = np.array(loss_test)
    plot_labels = [
        "pde_u",
        "pde_v",
        "pde_cont",
        "bc_wall_u_up",
        "bc_wall_u_down",
        "bc_wall_v_up",
        "bc_wall_v_down",
        "bc_inlet_u",
        "bc_inlet_v",
        "bc_outlet_p",
        "bc_outlet_v",
    ]
    # plot_array = np.array(losshistory.loss_train)
    for i in range(plot_array.shape[1]):
        plt.plot(loss_steps, plot_array[:, i], label=plot_labels[i])

    plt.legend()
    plt.ylim([0, 0.1])
    plt.savefig(f"./{simulation_name}/{plot_name}.png")
    # plt.show()
    plt.close()


def plot_probes(model, L, simulation_name: str):
    n_points = 1000
    x2 = np.linspace(-L / 2, L / 2, n_points)
    samples = np.zeros((n_points, 2))
    samples[:, 0] = x2

    result = model.predict(samples)

    fig, ax = plt.subplots()
    # ax.plot(samples[:, 1], u_exact, label="Analytic")
    # ax.plot(x2, result[:, 2], label="PINN (p)", linestyle="dashed", color="blue")
    ax.plot(
        x2,
        np.gradient(result[:, 0]),
        label="PINN (du/dx)",
        linestyle="dashed",
        color="blue",
    )
    ax.set_ylabel("(du/dx)", color="blue")

    ax2 = ax.twinx()
    ax2.plot(x2, result[:, 0], label="PINN (u)", linestyle="dashed", color="red")
    ax2.set_ylabel("(u)", color="red")
    # ax.legend()
    # ax2.legend()
    fig.savefig(f"./{simulation_name}/u_profiles.png")
    # fig.show()


def plot_pressure(model, L, D, mu, rho, u_in, simulation_name: str):
    n_points = 1000
    x2 = np.linspace(-L / 2, L / 2, n_points)
    samples = np.zeros((n_points, 2))
    samples[:, 0] = x2

    result = model.predict(samples)

    fig, ax = plt.subplots()
    # ax.plot(samples[:, 1], u_exact, label="Analytic")
    ax.plot(x2, result[:, 2], label="PINN (p)", linestyle="dashed")
    ax.set_ylabel("(p)", color="blue")

    ax2 = ax.twinx()
    ax2.plot(
        x2,
        np.gradient(result[:, 2]),
        label="PINN (dp/dx)",
        linestyle="dashed",
        color="red",
    )
    ax2.set_ylabel("(dp/dx)", color="red")
    # ax.legend()
    # ax2.legend()
    fig.savefig(f"./{simulation_name}/p_profiles.png")
    # fig.show()

    Dh = 2 * D
    Re = rho * u_in * Dh / mu
    # Le = 0.05 * Re * Dh
    Le = 5 * D
    print(Re, Le)

    Delta_p_L_analytic = 12 * u_in * mu / (D**3)
    result = model.predict(np.array([[L / 2 - 0.1, 0]]))

    print("delta_p/L analytic", Delta_p_L_analytic)
    print("delta_p/L PINN", result[:, 2] / 0.1)

    pd.DataFrame(
        {"Delta_p_analytic": Delta_p_L_analytic, "Delta_p_PINN": result[:, 2] / 0.1}
    ).to_csv(f"./{simulation_name}/delta_p.csv")


def plot_velocity_profile(model, pde, mu, v_i, D, L, simulation_name: str):
    n_points = 1000
    x2 = np.linspace(-D / 2, D / 2, n_points)
    samples = np.zeros((n_points, 2))
    samples[:, 1] = x2

    result = model.predict(samples)

    probe_points = np.array([[L / 2 - 0.01, 0], [L / 2 - 0.5, 0]])
    probe_pressao = model.predict(probe_points)

    # delta_p = probe_pressao[0, 2] - probe_pressao[1, 2]
    # length = probe_points[0, 0] - probe_points[1, 0]
    delta_p = 12
    length = 1

    def u_analytic(x):
        position = x[:, 1] - x[:, 1].min()
        return np.array(
            1
            / mu
            * (
                position**2 / 2 * delta_p / length
                + (mu * v_i / D - D / 2 * delta_p / length) * position
            ),
            dtype=np.float32,
        )

    u_exact = u_analytic(samples).reshape(-1)

    f = model.predict(samples, operator=pde)
    l2_diff_u = dde.metrics.l2_relative_error(u_exact, result[:, 0])
    residual = np.mean(np.absolute(f))

    result_df = pd.DataFrame({"residual": [residual], "l2_diff_u": [l2_diff_u]})
    result_df.to_csv(f"./{simulation_name}/result.csv")

    fig, ax = plt.subplots()
    ax.plot(samples[:, 1], u_exact, label="Analytic")
    ax.plot(samples[:, 1], result[:, 0], label="PINN", linestyle="dashed")
    ax.legend()
    ax.set_ylim([0, 1.5])
    fig.savefig(f"./{simulation_name}/u_profile_comparison.png")
    # fig.show()


def export_results(model, D, L, simulation_name):
    probe_lines = {
        "center_horizontal_line": np.array(
            [[(i - 500) * L / 1000, 0] for i in range(1001)], dtype=np.float32
        ),
        "center_vertical_line": np.array(
            [[0, (i - 500) * D / 1000] for i in range(1001)], dtype=np.float32
        ),
        "inlet_probe": np.array(
            [[-L / 2 + 0.01, (i - 500) * D / 1000] for i in range(1001)],
            dtype=np.float32,
        ),
        "outlet_probe": np.array(
            [[L / 2 - 0.01, (i - 500) * D / 1000] for i in range(1001)],
            dtype=np.float32,
        ),
    }

    target_probe_step = 0.1

    for i in range(int((L / 2 - 0.01) / target_probe_step)):
        probe_lines |= {
            f"probe_at_x_{-L/2 + 0.01 + (i + 1) * target_probe_step:.2f}": np.array(
                [
                    [-L / 2 + 0.01 + (i + 1) * target_probe_step, (i - 500) * D / 1000]
                    for i in range(1001)
                ],
                dtype=np.float32,
            )
        }

    for probe_name, probe_points in probe_lines.items():
        probe_result = model.predict(probe_points)
        probe_df = pd.DataFrame(
            {
                "Points_x": probe_points[:, 0],
                "Points_y": probe_points[:, 1],
                "u": probe_result[:, 0],
                "v": probe_result[:, 1],
                "p": probe_result[:, 2],
            }
        )
        probe_df.to_csv(f"./{simulation_name}/{probe_name}.csv")


def plot_fields(geom, model, L, D, simulation_name):
    color_legend = [[0, 1.5], [-0.3, 0.3], [0, 35]]
    samples = geom.random_points(500000)
    result = model.predict(samples)

    for idx, field in enumerate(["u", "v", "p"]):
        plt.figure(figsize=(20, 4))
        plt.scatter(samples[:, 0], samples[:, 1], c=result[:, idx], s=2)
        plt.colorbar()
        plt.clim([result[:, idx].min(), result[:, idx].max()])
        plt.xlim((-L / 2, L / 2))
        plt.ylim((-D / 2, D / 2))
        plt.tight_layout()
        plt.savefig(f"./{simulation_name}/{field}_colormap.png")
        # plt.show()
        plt.close()
