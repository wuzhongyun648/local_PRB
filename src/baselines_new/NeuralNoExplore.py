from .packages import *


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class NeuralNoExplore:
    def __init__(self, dim, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = 0.01

    def select(self, context):
        with torch.no_grad():
            mu = self.func(to_tensor(context)).view(-1)
        return int(torch.argmax(mu).item())

    def update(self, context, reward):
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        self.reward.append(float(reward))

    def train(self, t=None):
        if not self.reward:
            return 0
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
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
