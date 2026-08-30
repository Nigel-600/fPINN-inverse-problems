import torch
from fpinn import PINN
def train_pinn(
    model, t_collocation, t_data, u_data, x_ic, y_ic,  optimizer, verbose = True
):

    grad_outputs_u1 = torch.zeros(t_collocation.shape[0], 2)
    grad_outputs_u1[:, 0] = 1.0
    grad_outputs_u2 = torch.zeros(t_collocation.shape[0], 2)
    grad_outputs_u2[:, 1] = 1.0
    t0 = torch.zeros(1, 1, requires_grad=True)

    # --- History tracking ---
    history = {
        "epoch": [],
        "data_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "total_loss": [],
        "a": [], "b": [], "c": [], "d": [],
    }

    for epoch in range(10_001):
        optimizer.zero_grad()
        # --- Loss 1: Boundary Condition Loss ---
        u_bc_pred = model(x_ic)
        loss_bc = torch.mean((u_bc_pred - y_ic) ** 2)
        u_collocation = model(t_collocation)   # shape (N, 2)
        u1_prime_t = torch.autograd.grad(
            u_collocation, t_collocation,
            grad_outputs=grad_outputs_u1, create_graph=True
        )[0]
        u2_prime_t = torch.autograd.grad(
            u_collocation, t_collocation,
            grad_outputs=grad_outputs_u2, create_graph=True
        )[0]
        # --- t0 branch: build grad_outputs from u_ic's shape, not t0's ---
        u_ic = model(t0)                       # shape (1, 2)
        grad_outputs_u1_0 = torch.zeros_like(u_ic)
        grad_outputs_u1_0[:, 0] = 1.0
        grad_outputs_u2_0 = torch.zeros_like(u_ic)
        grad_outputs_u2_0[:, 1] = 1.0
        u1_prime_0 = torch.autograd.grad(
            u_ic, t0, grad_outputs=grad_outputs_u1_0, create_graph=True
        )[0]
        u2_prime_0 = torch.autograd.grad(
            u_ic, t0, grad_outputs=grad_outputs_u2_0, create_graph=True
        )[0]
        t = t_collocation  # shape (N,1)
        alpha = 1 / (1 + torch.exp(-model.rho))
        # alpha = 0.8
        gamma_term = torch.exp(torch.lgamma(torch.tensor(1 - alpha)))
        def caputo_from_channel(x0_ch, xprime0_ch, x_t_ch, xprime_t_ch):
            c2 = (3 / t**3) * (2*x0_ch + t*xprime0_ch - 2*x_t_ch + t*xprime_t_ch)
            c1 = (2 / t**2) * (-3*x0_ch - 2*t*xprime0_ch + 3*x_t_ch - t*xprime_t_ch)
            c0 = xprime0_ch.expand_as(t)
            return (
                c2 * t**(3 - alpha) / gamma_term * (1/(3-alpha) - 2/(2-alpha) + 1/(1-alpha))
                + c1 * t**(2 - alpha) / gamma_term * (1/(1-alpha) - 1/(2-alpha))
                + c0 * t**(1 - alpha) / gamma_term * (1/(1-alpha))
            )
        phys_params = {
            'a': torch.exp(model.log_a), 'b': torch.exp(model.log_b),
            'c': torch.exp(model.log_c), 'd': torch.exp(model.log_d)
        }
        phys_params = {k: (lambda t, v=v: v) for k, v in phys_params.items()}
        f_dict_nn = {
            "F1": lambda t, x, y: x * (phys_params['a'](t) - phys_params['b'](t) * y),
            "F2": lambda t, x, y: y * (phys_params['c'](t) * x - phys_params['d'](t))
        }
        D_alpha_u1 = caputo_from_channel(
            u_ic[:, 0:1], u1_prime_0, u_collocation[:, 0:1], u1_prime_t
        )
        D_alpha_u2 = caputo_from_channel(
            u_ic[:, 1:2], u2_prime_0, u_collocation[:, 1:2], u2_prime_t
        )
        pde_residual1 = D_alpha_u1.squeeze(-1) - f_dict_nn["F1"](t_collocation.squeeze(-1), u_collocation[:,0], u_collocation[:,1])
        pde_residual2 = D_alpha_u2.squeeze(-1) - f_dict_nn["F2"](t_collocation.squeeze(-1), u_collocation[:,0], u_collocation[:,1])
        pde_residual = torch.stack([pde_residual1, pde_residual2], dim=1)
        pde_loss = torch.mean(pde_residual**2, dim=0).sum()
        u_data_pred = model(t_data)
        data_residual = torch.cat((u_data[:,0], u_data[:,1]), dim=1) - u_data_pred
        data_loss = torch.mean(data_residual**2, dim=0).sum()
        # --- Total Loss Optimization ---
        total_loss = loss_bc + pde_loss + data_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # --- Store history every epoch ---
        history["epoch"].append(epoch)
        history["data_loss"].append(data_loss.item())
        history["pde_loss"].append(pde_loss.item())
        history["bc_loss"].append(loss_bc.item())
        history["total_loss"].append(total_loss.item())
        history["a"].append(phys_params['a'](0).item())
        history["b"].append(phys_params['b'](0).item())
        history["c"].append(phys_params['c'](0).item())
        history["d"].append(phys_params['d'](0).item())

        if epoch % 200 == 0:
            print(f"Epoch {epoch}:\n Data Loss = {data_loss}\n PDE Loss = {pde_loss.item():.4f}\n", f"BC Loss = {loss_bc.item():.4f}\n Total Loss = {total_loss.item():.4f}")
            print(f"{alpha}")
            print(f"{phys_params['a'](0)}")
            print(f"{phys_params['b'](0)}")
            print(f"{phys_params['c'](0)}")
            print(f"{phys_params['d'](0)}\n")