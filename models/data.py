from pathlib import Path

import numpy as np
from deepxde import DirichletBC
from deepxde.data import PDE
from deepxde.model import LossHistory, Model, TrainState
from deepxde.nn.tensorflow import FNN

from models.domains import NewtonianParallelPlane
from models.PINN import PINNParameters


class TrainingData:
    def __init__(
        self,
        domain: NewtonianParallelPlane,
        boundary_condition_list: list[DirichletBC],
        pinn_parameters: PINNParameters,
    ):
        self.data: PDE = PDE(
            domain.geometry,
            domain.get_pde(),
            boundary_condition_list,
            num_domain=pinn_parameters.domain_points,
            num_boundary=pinn_parameters.boundary_points,
            num_test=pinn_parameters.number_of_test,
        )


class NeuralNetworkModel:
    def __init__(self, data: PDE, net: FNN) -> None:
        self.model = Model(data, net)

    def compile_ADAM_model(self, learning_rate: float):
        self.model.compile("adam", lr=learning_rate)

    def compile_LBFGS_model(self):
        self.model.compile("L-BFGS")

    def train_model(
        self, iterations: int, display_frequency: int
    ) -> tuple[LossHistory, TrainState]:
        return self.model.train(iterations=iterations, display_every=display_frequency)

    def export_model(self, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save(output_path, protocol="backend")

    def predict(self, samples: np.ndarray):
        return self.model.predict(samples)

    def load_results(self):
        Model()
