import deepxde as dde


def create_pde(mu, rho, loss_weight):
    def pde(X, Y):
        du_x = dde.grad.jacobian(Y, X, i=0, j=0)
        du_y = dde.grad.jacobian(Y, X, i=0, j=1)
        dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
        dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
        dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
        dp_y = dde.grad.jacobian(Y, X, i=2, j=1)

        # Definições para newtoniano para componentes do tensor de tensão
        tau_xy = mu * (du_y + dv_x)
        tau_xx = 2 * mu * du_x
        tau_yx = mu * (dv_x + du_y)
        tau_yy = 2 * mu * dv_y

        # Derivadas dos tensores de tensão
        dtau_xx_x = dde.grad.jacobian(tau_xx, X, i=0, j=0)
        dtau_xy_x = dde.grad.jacobian(tau_xy, X, i=0, j=0)
        dtau_yx_y = dde.grad.jacobian(tau_yx, X, i=0, j=1)
        dtau_yy_y = dde.grad.jacobian(tau_yy, X, i=0, j=1)

        # Termos de advecção para as componentes u e v
        advec_u = Y[:, 0:1] * du_x + Y[:, 1:2] * du_y
        advec_v = Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y

        # Tensões na particula fluida
        tensao_x = dp_x - dtau_yx_y - dtau_xx_x
        tensao_y = dp_y - dtau_xy_x - dtau_yy_y

        # Equações de Navier-Stokes modificadas para incluir todos os tensores de tensão
        pde_u = rho * advec_u + tensao_x
        pde_v = rho * advec_v + tensao_y

        # Equação de continuidade
        pde_cont = du_x + dv_y

        return [pde_u, pde_v, loss_weight * pde_cont]

    return pde
