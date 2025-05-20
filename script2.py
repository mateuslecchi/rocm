import torch
import torch.nn as nn
import torch.optim as optim

def check_rocm_and_hip():
    if torch.cuda.is_available() and torch.version.hip is not None:
        print(f"ROCm is enabled. ROCm version: {torch.version.hip}")
        return True
    elif torch.cuda.is_available() and torch.version.hip is None:
        print("ROCm is enabled but HIP runtime is not available.")
        return True
    else:
        print("ROCm is not enabled.")
        return False

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_random_data(batch_size=32, device='cpu'):
    inputs = torch.randn(batch_size, 10).to(device)
    targets = torch.randn(batch_size, 1).to(device)
    return inputs, targets

def train_indefinitely(device):
    net = SimpleNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    iteration = 0
    try:
        while True:
            inputs, targets = generate_random_data(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if iteration % 10000 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")

            iteration += 1
    except KeyboardInterrupt:
        print("Training interrupted. Exiting...")

if __name__ == "__main__":
    device = 'cuda' if check_rocm_and_hip() else 'cpu'
    train_indefinitely(device)