class NewtonianFluid:
    def __init__(self, mu: float, density: float):
        self.mu = mu
        self.density = density


class PowerLawFluid:
    def __init__(self, K: float, n: float, density: float):
        self.K = K
        self.n = n
        self.density = density


class BinghamFluid:
    def __init__(self, mu: float, tau_0: float, density: float):
        self.mu = mu
        self.tau_0 = tau_0
        self.density = density


class HerschelBulkleyFluid:
    def __init__(self, K: float, n: float, tau_0: float, density: float):
        self.K = K
        self.n = n
        self.tau_0 = tau_0
        self.density = density
