from .packages import *


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiag:
    """Diagonal NeuralUCB following the original GitHub NeuralUCBDiag flow."""

    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.nu = nu
        self.lr = 0.01
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,), device=device)

    def _grad(self, output):
        self.func.zero_grad()
        output.backward(retain_graph=True)
        return torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])

    def select(self, context):
        tensor = to_tensor(context)
        mu = self.func(tensor).view(-1)
        g_list = []
        scores = []
        for fx in mu:
            g = self._grad(fx)
            g_list.append(g)
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g * g / self.U))
            scores.append(fx.item() + sigma.item())

        arm = int(np.argmax(scores))
        self.U += g_list[arm] * g_list[arm]
        return arm

    def update(self, context, reward):
        # Original GitHub appends inside train(context, reward). We append every
        # round here so the runner can enforce the PRB Appendix A.1 schedule.
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        self.reward.append(float(reward))

    def train(self, t=None):
        if not self.reward:
            return 0
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr, weight_decay=self.lamdba)
        index = np.arange(len(self.reward))
        np.random.shuffle(index)
        cnt = 0
        total_loss = 0.0
        while True:
            batch_loss = 0.0
            for idx in index:
                c = self.context_list[idx].to(device)
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(c) - r) ** 2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return total_loss / cnt
            if batch_loss / len(self.reward) <= 1e-3:
                return batch_loss / len(self.reward)
