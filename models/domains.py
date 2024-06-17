import deepxde as dde
import numpy as np
from deepxde import DirichletBC
from deepxde.gradients import jacobian

from models.boundary_conditions import BoundaryConditionBuilder
from models.rheology import NewtonianFluid


class NewtonianParallelPlane:
    def __init__(
        self,
        distance: float,
        length: float,
        inlet_velocity: float,
        rheology: NewtonianFluid,
    ):
        self.distance = distance
        self.length = length
        self.rheology = rheology
        self.inlet_velocity = inlet_velocity
        self.__setup_geometry()

    def __setup_geometry(self):
        self.geometry = dde.geometry.Rectangle(
            xmin=[-self.length / 2, -self.distance / 2],
            xmax=[self.length / 2, self.distance / 2],
        )

    def build_boundary_conditions(self) -> dict[str, DirichletBC]:
        inlet_bc = BoundaryConditionBuilder.inlet_boundary(
            geometry=self.geometry,
            inlet_velocity=self.inlet_velocity,
            length=self.length,
        )
        outlet_bc = BoundaryConditionBuilder.outlet_boundary(
            geometry=self.geometry,
            length=self.length,
        )
        wall_bc = BoundaryConditionBuilder.wall_boundary(
            geometry=self.geometry, length=self.length, distance=self.distance
        )

        return inlet_bc | outlet_bc | wall_bc

    def sample_points(self, number_of_points: int) -> np.ndarray:
        return self.geometry.random_points(number_of_points)

    def get_pde(self):
        def newtonian_pde(X, Y):
            du_x = jacobian(Y, X, i=0, j=0)
            du_y = jacobian(Y, X, i=0, j=1)
            dv_x = jacobian(Y, X, i=1, j=0)
            dv_y = jacobian(Y, X, i=1, j=1)
            dp_x = jacobian(Y, X, i=2, j=0)
            dp_y = jacobian(Y, X, i=2, j=1)

            # Definições para newtoniano para componentes do tensor de tensão
            tau_xy = self.rheology.mu * (du_y + dv_x)
            tau_xx = 2 * self.rheology.mu * du_x
            tau_yx = self.rheology.mu * (dv_x + du_y)
            tau_yy = 2 * self.rheology.mu * dv_y

            # Derivadas dos tensores de tensão
            dtau_xx_x = jacobian(tau_xx, X, i=0, j=0)
            dtau_xy_x = jacobian(tau_xy, X, i=0, j=0)
            dtau_yx_y = jacobian(tau_yx, X, i=0, j=1)
            dtau_yy_y = jacobian(tau_yy, X, i=0, j=1)

            # Termos de advecção para as componentes u e v
            advec_u = Y[:, 0:1] * du_x + Y[:, 1:2] * du_y
            advec_v = Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y

            # Tensões na particula fluida
            tensao_x = dp_x - dtau_yx_y - dtau_xx_x
            tensao_y = dp_y - dtau_xy_x - dtau_yy_y

            # Equações de Navier-Stokes modificadas para incluir todos os tensores de tensão
            pde_u = self.rheology.density * advec_u + tensao_x
            pde_v = self.rheology.density * advec_v + tensao_y

            # Equação de continuidade
            pde_cont = du_x + dv_y

            return [pde_u, pde_v, pde_cont]

        return newtonian_pde

    @property
    def ReynoldsNumber(self) -> float:
        return (
            self.rheology.density * self.inlet_velocity * self.length / self.rheology.mu
        )
