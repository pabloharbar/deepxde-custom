import pathlib
import sys

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

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

plt.style.use("tableau-colorblind10")
dde.config.set_parallel_scaling("weak")  # Nao paralelo

case_setup = [
    {
        "simulation_name": "Caso_0",
        "rho": "1",
        "mu": "1",
        "u_in": "1",
        "D": "1",
        "L": "1",
        "v_i": "1",
        "n_domain": "1",
        "n_boundary": "1",
        "loss_weight": "1",
    },
    {"name": "Caso_0"},
]


def main(*args):
    simulation_name = "training"
    pathlib.Path(f"./{simulation_name}").mkdir(exist_ok=True)

    rho = 1
    mu = 1
    u_in = 1
    D = 1
    L = 2
    v_i = 0

    # n_domain = 2500
    # n_boundary = 500
    # n_domain = 6250
    # n_boundary = 1000
    n_domain = 12500
    n_boundary = 1833
    loss_weight = 1

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

    position_df = pd.DataFrame(
        {"pos_x": data.train_x_all[:, 0], "pos_y": data.train_x_all[:, 1]}
    )
    position_df.to_csv(f"./{simulation_name}/train_points.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
