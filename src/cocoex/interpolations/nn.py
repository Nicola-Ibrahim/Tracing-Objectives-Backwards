import torch
import torch.nn as nn


class NNInterpolator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Input: alpha (scalar)
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # Output: interpolated solution
        )

    def forward(self, alpha):
        return self.net(alpha)

    def fit(self, alphas, pareto_set, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        alphas_tensor = torch.tensor(alphas, dtype=torch.float32).unsqueeze(1)
        pareto_tensor = torch.tensor(pareto_set, dtype=torch.float32)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self(alphas_tensor)
            loss = loss_fn(outputs, pareto_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, alpha):
        with torch.no_grad():
            return self(torch.tensor([[alpha]], dtype=torch.float32)).numpy()
