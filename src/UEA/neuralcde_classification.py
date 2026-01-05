import torch
import torchcde

# Code adapted from : https://github.com/patrick-kidger/NeuralCDE
class F(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=2, mlp_hidden_dim=128):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        layers = []
        in_dim = hidden_channels
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = mlp_hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels * input_channels))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t, z):
        out = self.net(z)
        return out.view(z.shape[0], self.hidden_channels, self.input_channels)


class NCDE_classification(torch.nn.Module):
    """
    Paper-like NCDE:
    - input: X_points of shape (batch, length, channels)
    - build cubic spline coefficients internally
    - solve CDE with rk4
    - readout at final time
    """
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, mlp_hidden_dim=128, method="rk4"):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.method = method

        self.func = F(input_channels, hidden_channels, num_layers=num_layers, mlp_hidden_dim=mlp_hidden_dim)
        self.initial_linear = torch.nn.Linear(input_channels, hidden_channels)
        self.readout_linear = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, X_points):
        """
        X_points: (batch, length, channels)
        """
        # Build spline
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_points)
        X = torchcde.CubicSpline(coeffs)

        # Initial hidden state
        X0 = X.evaluate(X.interval[0])           # (batch, channels)
        z0 = self.initial_linear(X0)             # (batch, hidden)

        # Integrate on the natural interval
        zt = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=X.interval,
            method=self.method
        )                                        # (batch, len(t), hidden)

        zT = zt[:, -1, :]                        # final hidden state
        return self.readout_linear(zT)           # logits