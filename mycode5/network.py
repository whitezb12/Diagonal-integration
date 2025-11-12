import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()        
        n_hidden = 512
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)
        z = self.fc2(h)
        return z

class Generator(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_output: int,
        loss_type: str = "MSE",
    ) -> None:
        super().__init__()        
        n_hidden = 512
        self.loss_type = loss_type
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = F.relu(h)
        out = self.fc2(h)
        if self.loss_type == "BCE":
            out = torch.sigmoid(out)
        return out


class BinaryDiscriminator(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)


class MultiClassDiscriminator(nn.Module):
    def __init__(self, n_input: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)