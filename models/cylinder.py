import itertools
from typing import Literal

import numpy as np
from deepxde import backend as bkd
from deepxde import config
from deepxde.geometry import Geometry
from deepxde.geometry.sampler import sample
from deepxde.utils import isclose
from scipy import stats
from sklearn import preprocessing


class HyperCylinder(Geometry):
    def __init__(
        self, center: tuple[float, float, float], radius: float, height: float
    ):
        self.center = np.array(center, dtype=config.real(np))
        self.radius = radius
        self.height = height
        print(self.center)
        super().__init__(
            dim=3,  # Dimensões: x, y para a base, z para a altura
            bbox=(
                np.array([center[0] - radius, center[1] - radius, 0]),  # mínimo x, y, z
                np.array(
                    [center[0] + radius, center[1] + radius, height]
                ),  # máximo x, y, z
            ),
            diam=np.sqrt(2 * radius**2 + height**2),  # Diagonal do cilindro
        )
        self._r2 = radius**2  # Otimização para cálculos de ponto dentro do cilindro

    def inside(self, x):
        # Verifica se o ponto está dentro do raio e entre 0 e a altura do cilindro no eixo z.
        radial_distance = np.sqrt(
            (x[:, 0] - self.center[0]) ** 2 + (x[:, 1] - self.center[1]) ** 2
        )
        return (
            (radial_distance <= self.radius) & (x[:, 2] >= 0) & (x[:, 2] <= self.height)
        )

    def on_boundary(self, x):
        radial_distance = np.sqrt(
            (x[:, 0] - self.center[0]) ** 2 + (x[:, 1] - self.center[1]) ** 2
        )
        on_side = isclose(radial_distance, self.radius)
        on_top = isclose(x[:, 2], self.height)
        on_bottom = isclose(x[:, 2], 0)
        return on_side | on_top | on_bottom

    def distance2boundary_unitdirn(self, x, dirn):
        # Ajustes para o centro e raio na componente xy
        xc_xy = x[:2] - self.center[:2]
        dirn_xy = dirn[:2]

        # Calcula os coeficientes do problema quadrático
        a = np.dot(dirn_xy, dirn_xy)
        b = 2 * np.dot(xc_xy, dirn_xy)
        c = np.dot(xc_xy, xc_xy) - self.radius**2

        # Resolve o problema quadrático para encontrar t
        delta = b**2 - 4 * a * c
        if delta < 0:
            return np.inf  # Sem intersecção
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)

        # Retorna a menor distância positiva
        return min(t for t in [t1, t2] if t > 0)

    def boundary_constraint_factor(
        self, x, smoothness: Literal["C0", "C0+", "Cinf"] = "C0+"
    ):
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")

        # Convertendo o centro e o raio para tensores se ainda não foram convertidos
        if not hasattr(self, "center_tensor"):
            self.center_tensor = bkd.as_tensor(
                self.center[:2]
            )  # Apenas coordenadas x, y
            self.radius_tensor = bkd.as_tensor(self.radius)

        # Calculando a distância radial do ponto até o eixo central do cilindro
        radial_distance = (
            bkd.norm(x[:, :2] - self.center_tensor, axis=-1, keepdims=True)
            - self.radius_tensor
        )

        # Distâncias para as superfícies superior e inferior
        distance_top = bkd.abs(x[:, 2:3] - self.height)
        distance_bottom = bkd.abs(x[:, 2:3])

        # Selecionando a menor distância entre lateral, topo e fundo
        dist = bkd.minimum(radial_distance, bkd.minimum(distance_top, distance_bottom))

        # Aplicando suavização conforme especificado
        if smoothness == "Cinf":
            dist = bkd.square(dist)
        else:
            dist = bkd.abs(dist)

        return dist

    def boundary_normal(self, x):
        # Vetor do ponto ao centro projetado no plano xy
        _n = x[:, :2] - self.center[:2]
        radial_distance = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = (_n / radial_distance) * isclose(radial_distance, self.radius)

        # Adiciona a verificação para as faces superior e inferior
        on_top = isclose(x[:, 2], self.height)
        on_bottom = isclose(x[:, 2], 0)
        _n = np.where(on_top, np.array([0, 0, 1]), _n)
        _n = np.where(on_bottom, np.array([0, 0, -1]), _n)

        return _n

    def random_points(self, n, random="pseudo"):
        if random == "pseudo":
            # Gera raios uniformemente distribuídos na área do círculo da base
            radii = np.sqrt(np.random.rand(n, 1)) * self.radius
            # Gera ângulos uniformemente distribuídos
            angles = np.random.rand(n, 1) * 2 * np.pi
            # Coordenadas x e y no círculo da base
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            # Alturas uniformemente distribuídas
            z = np.random.rand(n, 1) * self.height
        else:
            # Para um gerador de números aleatórios mais sofisticado ou para teste
            rng = sample(n, 3, random)  # Considerando dim = 3 para x, y, z
            x = np.sqrt(rng[:, 0:1]) * self.radius * np.cos(2 * np.pi * rng[:, 1:2])
            y = np.sqrt(rng[:, 0:1]) * self.radius * np.sin(2 * np.pi * rng[:, 1:2])
            z = rng[:, 2:3] * self.height

        # Combina as coordenadas x, y, e z
        points = np.hstack((x, y, z))
        # Desloca os pontos para o centro correto
        points += np.array([self.center[0], self.center[1], 0])

        return points.astype(config.real(np))

    def random_boundary_points(self, n, random="pseudo"):
        if random == "pseudo":
            # Gerar pontos para a superfície lateral
            angles = np.random.rand(n, 1) * 2 * np.pi
            heights = np.random.rand(n, 1) * self.height
            lateral_x = self.radius * np.cos(angles) + self.center[0]
            lateral_y = self.radius * np.sin(angles) + self.center[1]
            lateral_z = heights

            # Gerar pontos para as bases (topo e fundo)
            n_base = n // 3  # Um terço dos pontos para cada base
            angles_base = np.random.rand(2 * n_base, 1) * 2 * np.pi
            radii_base = np.sqrt(np.random.rand(2 * n_base, 1)) * self.radius
            base_x = radii_base * np.cos(angles_base)
            base_y = radii_base * np.sin(angles_base)
            base_z_top = np.full((n_base, 1), self.height)
            base_z_bottom = np.zeros((n_base, 1))

            # Concatena e mescla os pontos de todas as partes
            boundary_points_x = np.concatenate(
                (lateral_x, base_x[:n_base], base_x[n_base:])
            )
            boundary_points_y = np.concatenate(
                (lateral_y, base_y[:n_base], base_y[n_base:])
            )
            boundary_points_z = np.concatenate((lateral_z, base_z_top, base_z_bottom))

            points = np.hstack(
                (boundary_points_x, boundary_points_y, boundary_points_z)
            )
        else:
            # Para geradores de números aleatórios avançados ou personalizados
            U = sample(n, 3, random)  # Adapte conforme necessário
            points = preprocessing.normalize(stats.norm.ppf(U).astype(config.real(np)))
            points = self.radius * points
            points[:, 2] = (
                points[:, 2] % self.height
            )  # Ajusta a altura para ficar dentro dos limites do cilindro

        return points + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = (
            x
            - np.arange(-shift, n - shift + 1, dtype=config.real(np))[:, None]
            * h
            * dirn
        )
        return pts


class Cylinder(HyperCylinder):
    """
    Args:
        center: Center of the cylinder.
        radius: Radius of the cylinder.
        height: Height of the cylinder.
    """
