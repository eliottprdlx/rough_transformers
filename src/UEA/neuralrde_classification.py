import signatory
import torch
import torch.nn as nn
import rdeint

# Code adapted from : https://github.com/fmorenopino/neuralRDE.
class F(nn.Module):
    def __init__(self, input_channels, logsignature_channels, hidden_channels, num_layers=2, mlp_hidden_dim=128):
        super().__init__()
        self.input_channels = input_channels
        self.logsignature_channels = logsignature_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        layers = []
        in_dim = hidden_channels

        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = mlp_hidden_dim

        layers.append(
            torch.nn.Linear(
                in_dim,
                input_channels * logsignature_channels
            )
        )

        self.net = torch.nn.Sequential(*layers)

    def forward(self, t, z):
        # z: (batch, hidden_channels)
        out = self.net(z)
        return out.view(
            -1,
            self.input_channels,
            self.logsignature_channels,
        ) 

class NRDE_classification(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, logsignature_depth):
        super(NRDE_classification, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.logsignature_depth = logsignature_depth
        self.logsignature_channels = signatory.logsignature_channels(
            channels=input_channels,
            depth=logsignature_depth
        )
        self.func = F(
            input_channels,
            self.logsignature_channels,
            hidden_channels
        )
        self.initial_linear = torch.nn.Linear(self.input_channels, self.hidden_channels)
        self.readout_linear = torch.nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, X):
        # Compute log-signature of the path X
        logsig = signatory.logsignature(
            X.evaluate_intervals(),
            depth=self.logsignature_depth
        )

        # Initial hidden state
        X0 = X.evaluate(X.interval[0])
        h0 = self.initial_linear(X0)

        # Solve the RDE
        sol = rdeint.rdeint(
            logsig=logsig,
            h0=h0,
            func=self.func,
            t=X.interval,
            method='rk4'
        )

        h_T = sol[:, -1, :]  # Take the final time step

        # Readout
        out = self.readout_linear(h_T)
        return out