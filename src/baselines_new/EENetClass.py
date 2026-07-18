from skimage.measure import block_reduce

from .packages import *


class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class Network_exploration(nn.Module):
    def __init__(self, input_dim, _kernel_size=100, _stride=50, _channels=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, _channels, kernel_size=_kernel_size, stride=_stride)
        num_dim = int(((input_dim - _kernel_size) / _stride + 1) * _channels)
        if num_dim <= 0:
            raise ValueError(
                f"EE-Net exploration input_dim={input_dim} is smaller than kernel_size={_kernel_size}"
            )
        self.fc1 = nn.Linear(num_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.activate(self.fc1(x))
        return self.fc2(x)


class Network_decision_maker(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class Exploitation:
    def __init__(self, input_dim, num_arm, pool_step_size, lr=0.01, hidden=100):
        self.func = Network_exploitation(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr
        self.pool_step_size = pool_step_size
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

    def update(self, context, reward):
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        self.reward.append(float(reward))

    def output_and_gradient(self, context):
        tensor = to_tensor(context)
        results = self.func(tensor)
        g_list = []
        res_list = []
        for fx in results:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(np.array(g.cpu()))
            res_list.append([fx.item()])
        g_list = block_reduce(np.array(g_list), block_size=(1, self.pool_step_size), func=np.mean)
        return np.array(res_list), g_list

    def train(self):
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
                torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return total_loss / cnt
            if batch_loss / len(self.reward) <= 1e-3:
                return batch_loss / len(self.reward)


class Exploration:
    def __init__(self, input_dim, lr=0.001, kernel_size=100):
        self.func = Network_exploration(input_dim=input_dim, _kernel_size=kernel_size).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr

    def update(self, context, reward):
        tensor = torch.as_tensor(context, dtype=torch.float32, device=device)
        tensor = torch.unsqueeze(torch.unsqueeze(tensor, 0), 0)
        self.context_list.append(tensor)
        self.reward.append(float(reward))

    def output(self, context):
        tensor = torch.unsqueeze(to_tensor(context), 1)
        with torch.no_grad():
            return self.func(tensor).cpu().numpy()

    def train(self):
        if not self.reward:
            return 0
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr, weight_decay=0.0001)
        index = np.arange(len(self.reward))
        np.random.shuffle(index)
        cnt = 0
        total_loss = 0.0
        while True:
            batch_loss = 0.0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(c.to(device)) - r) ** 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return total_loss / cnt
            if batch_loss / len(self.reward) <= 2e-3:
                return batch_loss / len(self.reward)


class Decision_maker:
    def __init__(self, input_dim, hidden=20, lr=0.01):
        self.func = Network_decision_maker(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr

    def update(self, context, reward):
        self.context_list.append(torch.as_tensor(context.reshape(1, -1), dtype=torch.float32))
        self.reward.append(float(reward))

    def select(self, context):
        with torch.no_grad():
            res = self.func(to_tensor(context)).cpu().numpy()
        return int(np.argmax(res))

    def train(self):
        if not self.reward:
            return 0
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
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
                if cnt >= 1000:
                    return total_loss / cnt
            if batch_loss / len(self.reward) <= 1e-3:
                return batch_loss / len(self.reward)
