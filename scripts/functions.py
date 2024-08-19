import numpy as np


def create_upper_wall(D, L):
    def upper_wall(X, on_boundary):
        on_upper_wall = np.logical_and(
            np.logical_and(
                np.isclose(X[1], D / 2, rtol=1e-05, atol=1e-08),
                np.not_equal(X[0], -L / 2),
            ),
            on_boundary,
        )
        return on_upper_wall

    return upper_wall


def create_lower_wall(D, L):
    def lower_wall(X, on_boundary):
        on_lower_wall = np.logical_and(
            np.logical_and(
                np.isclose(X[1], -D / 2, rtol=1e-05, atol=1e-08),
                np.not_equal(X[0], -L / 2),
            ),
            on_boundary,
        )
        return on_lower_wall

    return lower_wall


def create_boundary_inlet(L):
    def boundary_inlet(X, on_boundary):
        on_inlet = np.logical_and(
            np.isclose(X[0], -L / 2, rtol=1e-05, atol=1e-08), on_boundary
        )
        return on_inlet

    return boundary_inlet


def create_boundary_outlet(L):
    def boundary_outlet(X, on_boundary):
        on_outlet = np.logical_and(
            np.isclose(X[0], L / 2, rtol=1e-05, atol=1e-08), on_boundary
        )
        return on_outlet

    return boundary_outlet


def zero(x):
    return 0.0


def create_inlet_u(u_in):
    def inlet_u(x):
        return u_in

    return inlet_u


def create_wall_u(v_i):
    def wall_u(x):
        return v_i

    return wall_u
