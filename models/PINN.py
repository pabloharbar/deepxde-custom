from dataclasses import dataclass


@dataclass
class PINNParameters:
    domain_points: int
    boundary_points: int
    number_of_test: int
