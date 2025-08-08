import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class DomainWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        domain: Literal['A', 'B'],
        use_prefix: bool = True,
        use_domain_bn: bool = True,
    ) -> None:
        super().__init__()
        assert domain in ['A', 'B']
        self.model = base_model
        self.domain = domain
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn

        if use_prefix:
            prefix = torch.tensor([1., 0.]) if domain == 'A' else torch.tensor([0., 1.])
            self.register_buffer("prefix_tensor", prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_prefix:
            batch_size = x.size(0)
            prefix = self.prefix_tensor.expand(batch_size, 2).to(x.device)
            x = torch.cat([prefix, x], dim=1)

        if self.use_domain_bn:
            return self.model(x, domain=self.domain)
        else:
            return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        use_prefix: bool = False,
        use_domain_bn: bool = False,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn

        n_hidden = 512
        input_dim = n_input + 2 if use_prefix else n_input

        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_logvar = nn.Linear(n_hidden, n_latent)

        if use_domain_bn:
            self.bn_A = nn.BatchNorm1d(n_hidden)
            self.bn_B = nn.BatchNorm1d(n_hidden)

    def forward(self, x: torch.Tensor, domain: Optional[Literal['A', 'B']] = None) -> torch.Tensor:
        h = self.fc1(x)

        if self.use_domain_bn:
            if domain == 'A':
                h = self.bn_A(h)
            elif domain == 'B':
                h = self.bn_B(h)
            else:
                raise ValueError("Domain must be 'A' or 'B'")

        h = F.relu(h)
        h = self.dropout(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z


class Generator(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        loss_type: Literal['MSE', 'BCE'] = 'MSE',
        use_prefix: bool = False,
        use_domain_bn: bool = False,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn

        n_hidden = 512
        input_dim = n_latent + 2 if use_prefix else n_latent

        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_input)

        if use_domain_bn:
            self.bn_A = nn.BatchNorm1d(n_hidden)
            self.bn_B = nn.BatchNorm1d(n_hidden)

    def forward(self, z: torch.Tensor, domain: Optional[Literal['A', 'B']] = None) -> torch.Tensor:
        h = self.fc1(z)

        if self.use_domain_bn:
            if domain == 'A':
                h = self.bn_A(h)
            elif domain == 'B':
                h = self.bn_B(h)
            else:
                raise ValueError("Domain must be 'A' or 'B'")

        h = F.relu(h)
        x = self.fc2(h)

        if self.loss_type == 'BCE':
            x = torch.sigmoid(x)

        return x


class BinaryDiscriminator(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)


class MultiClassDiscriminator(nn.Module):
    def __init__(self, n_input: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
