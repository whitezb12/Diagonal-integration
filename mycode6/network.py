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
        n_output: int
    ) -> None:
        super().__init__()        
        n_hidden = 512
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = F.relu(h)
        out = self.fc2(h)
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
"""
n_hidden = 512


def get_norm_layer(norm_type: Literal["batchnorm", "layernorm", "none"], dim: int):
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(dim)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_activation(act_type: Literal["relu", "leakyrelu", "gelu", "elu", "silu", "none"]):
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(0.2)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "elu":
        return nn.ELU()
    elif act_type == "silu":
        return nn.SiLU()
    else:
        return nn.Identity()


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        dropout_rate: float = 0.2,
        norm_type: Literal["batchnorm", "layernorm", "none"] = "none",
        act_type: Literal["relu", "leakyrelu", "gelu", "elu", "silu", "none"] = "relu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.norm = get_norm_layer(norm_type, n_hidden)
        self.act = get_activation(act_type)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        z = self.fc2(h)
        return z


class VAEEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        dropout_rate: float = 0.2,
        norm_type: Literal["batchnorm", "layernorm", "none"] = "none",
        act_type: Literal["relu", "leakyrelu", "gelu", "elu", "silu", "none"] = "relu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.norm = get_norm_layer(norm_type, n_hidden)
        self.act = get_activation(act_type)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor):
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Generator(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_output: int,
        norm_type: Literal["batchnorm", "layernorm", "none"] = "none",
        act_type: Literal["relu", "leakyrelu", "gelu", "elu", "silu", "none"] = "relu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.norm = get_norm_layer(norm_type, n_hidden)
        self.act = get_activation(act_type)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = self.norm(h)
        h = self.act(h)
        out = self.fc2(h)
        return out


class BinaryDiscriminator(nn.Module):
    def __init__(
        self,
        n_input: int,
        norm_type: Literal["batchnorm", "layernorm", "none"] = "none",
        act_type: Literal["relu", "leakyrelu", "gelu", "elu", "silu", "none"] = "relu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            get_norm_layer(norm_type, n_hidden),
            get_activation(act_type),
            nn.Linear(n_hidden, n_hidden),
            get_norm_layer(norm_type, n_hidden),
            get_activation(act_type),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)


class MultiClassDiscriminator(nn.Module):
    def __init__(
        self,
        n_input: int,
        num_classes: int,
        norm_type: Literal["batchnorm", "layernorm", "none"] = "none",
        act_type: Literal["relu", "leakyrelu", "gelu", "elu", "silu", "none"] = "relu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            get_norm_layer(norm_type, n_hidden),
            get_activation(act_type),
            nn.Linear(n_hidden, n_hidden),
            get_norm_layer(norm_type, n_hidden),
            get_activation(act_type),
            nn.Linear(n_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)
"""