from .packages import *


class KernelUCB:
    def __init__(self, dim, lamdba=1, nu=1):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.x_t = None
        self.r_t = None
        self.history_len = 0
        self.scale = self.lamdba * self.nu
        self.U_t = None
        self.K_t = None

    def select(self, context):
        a, _ = context.shape
        if self.history_len == 0:
            mu_t = torch.zeros((a,), device=device)
            sigma_t = self.scale * torch.ones((a,), device=device)
        else:
            c_t = to_tensor(context)
            delta_t = c_t.reshape((a, 1, -1)) - self.x_t.reshape((1, self.history_len, -1))
            k_t = torch.exp(-delta_t.norm(dim=2))
            mu_t = k_t.matmul(self.U_t.matmul(self.r_t))
            sigma_t = self.scale * (
                torch.ones((a,), device=device)
                - torch.diag(k_t.matmul(self.U_t.matmul(k_t.T)))
            )
            sigma_t = torch.clamp(sigma_t, min=1e-12)

        r = mu_t + torch.sqrt(sigma_t)
        return int(torch.argmax(r).item())

    def train(self, context, reward):
        if self.history_len < 1000:
            if self.x_t is None:
                self.x_t = to_tensor(context).reshape((1, -1))
                self.r_t = torch.tensor(reward, device=device, dtype=torch.float32).reshape((-1,))
                self.K_t = torch.ones((1, 1), device=device, dtype=torch.float32)
            else:
                c_t = to_tensor(context).reshape((1, -1))
                r_t = torch.tensor(reward, device=device, dtype=torch.float32).reshape((-1,))
                delta_t = c_t.reshape((1, 1, -1)) - self.x_t.reshape((1, self.history_len, -1))
                self.x_t = torch.cat((self.x_t, c_t), dim=0)
                self.r_t = torch.cat((self.r_t, r_t), dim=0)
                k_t = torch.exp(-delta_t.norm(dim=2)).reshape((-1, 1))
                a = torch.cat((k_t.T, torch.ones((1, 1), dtype=torch.float32, device=device)), dim=1)
                b = torch.cat((self.K_t, k_t), dim=1)
                self.K_t = torch.cat((b, a), dim=0)
            self.history_len += 1
            eye = torch.eye(self.history_len, device=device)
            self.U_t = torch.inverse(self.K_t + self.lamdba * eye)
        return 0
