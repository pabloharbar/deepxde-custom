from dataclasses import dataclass
from typing import Literal

from deepxde.nn.tensorflow import FNN


@dataclass
class ADAMOptimizer:
    iterations: int
    report_frequency: int
    learning_rate: float = 1e-3


@dataclass
class LBFGSOptimizer:
    max_iterations: int
    report_frequency: int


@dataclass
class NeuralNetwork:
    adam_optimizer: ADAMOptimizer
    LBFGS_optimizer: LBFGSOptimizer
    input_layer_neurons: int = 2
    hidden_layer_neurons: int = 64
    hidden_layer_number: int = 5
    output_layer_neurons: int = 3
    activation_function: Literal["tanh", "sigmoid"] = "tanh"
    point_distribution: Literal["Glorot uniform"] = "Glorot uniform"

    def build_nn_list(self) -> list[int]:
        return (
            [self.input_layer_neurons]
            + [self.hidden_layer_neurons] * self.hidden_layer_number
            + [self.output_layer_neurons]
        )

    def build_net(self) -> FNN:
        return FNN(
            self.build_nn_list(), self.activation_function, self.point_distribution
        )
