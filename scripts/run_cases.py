import pathlib

import deepxde as dde
import pandas as pd

from scripts.functions import (
    create_boundary_inlet,
    create_boundary_outlet,
    create_inlet_u,
    create_lower_wall,
    create_upper_wall,
    create_wall_u,
    zero,
)
from scripts.pde import create_pde
from scripts.plotting import (
    export_results,
    plot_fields,
    plot_loss_history,
    plot_pressure,
    plot_probes,
    plot_training_points,
    plot_velocity_profile,
)

dde.config.set_parallel_scaling("weak")  # Nao paralelo

case_setup = [
    {
        "simulation_name": "Caso_0",
        "rho": 1,
        "mu": 1,
        "u_in": 1,
        "D": 1,
        "L": 2,
        "v_i": 0,
        "n_domain": 2500,
        "n_boundary": 500,
        "loss_weight": 1,
    },
    {
        "simulation_name": "Caso_1",
        "rho": 1,
        "mu": 1,
        "u_in": 1,
        "D": 1,
        "L": 5,
        "v_i": 0,
        "n_domain": 6250,
        "n_boundary": 1000,
        "loss_weight": 1,
    },
    {
        "simulation_name": "Caso_2",
        "rho": 1,
        "mu": 1,
        "u_in": 1,
        "D": 1,
        "L": 10,
        "v_i": 0,
        "n_domain": 12500,
        "n_boundary": 1833,
        "loss_weight": 1,
    },
]


def main():
    for case_dict in case_setup:
        print(f"Starting case {case_dict['simulation_name']}")
        simulation_name = case_dict["simulation_name"]
        pathlib.Path(f"./{simulation_name}").mkdir(exist_ok=True)

        rho = case_dict["rho"]
        mu = case_dict["mu"]
        u_in = case_dict["u_in"]
        D = case_dict["D"]
        L = case_dict["L"]
        v_i = case_dict["v_i"]

        n_domain = case_dict["n_domain"]
        n_boundary = case_dict["n_boundary"]
        loss_weight = case_dict["loss_weight"]

        Re = rho * u_in * D / mu
        geom = dde.geometry.Rectangle(xmin=[-L / 2, -D / 2], xmax=[L / 2, D / 2])

        upper_wall = create_upper_wall(D, L)
        lower_wall = create_lower_wall(D, L)
        boundary_inlet = create_boundary_inlet(L)
        boundary_outlet = create_boundary_outlet(L)
        wall_u = create_wall_u(v_i)
        inlet_u = create_inlet_u(u_in)

        bc_wall_u_up = dde.DirichletBC(geom, wall_u, upper_wall, component=0)
        bc_wall_u_down = dde.DirichletBC(geom, zero, lower_wall, component=0)
        bc_wall_v_up = dde.DirichletBC(geom, zero, upper_wall, component=1)
        bc_wall_v_down = dde.DirichletBC(geom, zero, lower_wall, component=1)

        bc_inlet_u = dde.DirichletBC(geom, inlet_u, boundary_inlet, component=0)
        bc_inlet_v = dde.DirichletBC(geom, zero, boundary_inlet, component=1)

        bc_outlet_p = dde.DirichletBC(geom, zero, boundary_outlet, component=2)
        bc_outlet_v = dde.DirichletBC(geom, zero, boundary_outlet, component=1)

        pde = create_pde(mu, rho, loss_weight)
        data = dde.data.PDE(
            geom,
            pde,
            [
                bc_wall_u_up,
                bc_wall_u_down,
                bc_wall_v_up,
                bc_wall_v_down,
                bc_inlet_u,
                bc_inlet_v,
                bc_outlet_p,
                bc_outlet_v,
            ],
            num_domain=n_domain,
            num_boundary=n_boundary,
            num_test=2000,
        )
        position_df = pd.read_csv(f"./training/{n_domain}-{n_boundary}.csv")
        data.train_x_all[:, 0] = position_df["pos_x"].to_numpy()
        data.train_x_all[:, 1] = position_df["pos_y"].to_numpy()

        plot_training_points(
            data.train_x_all[:, 0], data.train_x_all[:, 1], simulation_name
        )
        print(f"Finished setup case {case_dict['simulation_name']}")

        net = dde.maps.FNN([2] + [64] * 5 + [3], "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        # model.compile("adam", lr=1e-3, loss_weights=[1,1,loss_weight,1,1,1,1,1,1,1,1])
        model.compile("adam", lr=1e-3)
        losshistory, train_state = model.train(iterations=4000, display_every=100)
        plot_loss_history(
            losshistory.loss_test, losshistory.steps, simulation_name, "residual_adam"
        )
        print(f"Finished adam optimizer case {case_dict['simulation_name']}")

        dde.optimizers.config.set_LBFGS_options(maxiter=2500)
        model.compile("L-BFGS")
        losshistory, train_state = model.train(iterations=3000, display_every=100)
        dde.saveplot(losshistory, train_state, issave=False, isplot=False)
        plot_loss_history(
            losshistory.loss_test, losshistory.steps, simulation_name, "residual_L-BFGS"
        )
        print(f"Finished L-BFGS optimizer case {case_dict['simulation_name']}")
        plot_probes(model, L, simulation_name)
        plot_pressure(model, L, D, mu, rho, u_in, simulation_name)
        plot_velocity_profile(model, pde, mu, v_i, D, L, simulation_name)
        export_results(model, D, L, simulation_name)
        plot_fields(geom, model, L, D, simulation_name)
        print(f"Finished exporting case {case_dict['simulation_name']}")


if __name__ == "__main__":
    main()
