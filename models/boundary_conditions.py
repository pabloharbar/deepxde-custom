import numpy as np
from deepxde import DirichletBC
from deepxde.geometry import Rectangle


class BoundaryConditionBuilder:
    @classmethod
    def inlet_boundary(
        self, geometry: Rectangle, inlet_velocity: float, length: float
    ) -> dict[str, DirichletBC]:
        def boundary_inlet(X, on_boundary):
            on_inlet = np.logical_and(
                np.isclose(X[0], -length / 2, rtol=1e-05, atol=1e-08),
                on_boundary,
            )
            return on_inlet

        bc_inlet_u = DirichletBC(
            geometry, lambda X: inlet_velocity, boundary_inlet, component=0
        )
        bc_inlet_v = DirichletBC(geometry, lambda X: 0.0, boundary_inlet, component=1)

        return {"bc_inlet_u": bc_inlet_u, "bc_inlet_v": bc_inlet_v}

    @classmethod
    def outlet_boundary(
        self, geometry: Rectangle, length: float
    ) -> dict[str, DirichletBC]:
        def boundary_outlet(X, on_boundary):
            on_outlet = np.logical_and(
                np.isclose(X[0], length / 2, rtol=1e-05, atol=1e-08), on_boundary
            )
            return on_outlet

        bc_outlet_p = DirichletBC(geometry, lambda X: 0.0, boundary_outlet, component=2)
        bc_outlet_v = DirichletBC(geometry, lambda X: 0.0, boundary_outlet, component=1)

        return {"bc_outlet_p": bc_outlet_p, "bc_outlet_v": bc_outlet_v}

    @classmethod
    def wall_boundary(
        self, geometry: Rectangle, length: float, distance: float
    ) -> dict[str, DirichletBC]:
        def upper_wall(X, on_boundary):
            on_upper_wall = np.logical_and(
                np.logical_and(
                    np.isclose(X[1], distance / 2, rtol=1e-05, atol=1e-08),
                    np.not_equal(X[0], -length / 2),
                ),
                on_boundary,
            )
            return on_upper_wall

        def lower_wall(X, on_boundary):
            on_lower_wall = np.logical_and(
                np.logical_and(
                    np.isclose(X[1], -distance / 2, rtol=1e-05, atol=1e-08),
                    np.not_equal(X[0], -length / 2),
                ),
                on_boundary,
            )
            return on_lower_wall

        bc_wall_u_up = DirichletBC(geometry, lambda X: 0.0, upper_wall, component=0)
        bc_wall_u_down = DirichletBC(geometry, lambda X: 0.0, lower_wall, component=0)
        bc_wall_v_up = DirichletBC(geometry, lambda X: 0.0, upper_wall, component=1)
        bc_wall_v_down = DirichletBC(geometry, lambda X: 0.0, lower_wall, component=1)

        return {
            "bc_wall_u_up": bc_wall_u_up,
            "bc_wall_u_down": bc_wall_u_down,
            "bc_wall_v_up": bc_wall_v_up,
            "bc_wall_v_down": bc_wall_v_down,
        }
