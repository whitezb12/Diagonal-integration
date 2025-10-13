import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple


class BaseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_prefix: bool = False) -> None:
        super().__init__()
        self.use_prefix = use_prefix
        self.fc = nn.Linear(in_dim + (2 if use_prefix else 0), out_dim)

    def forward(self, x: torch.Tensor, prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_prefix and prefix is not None:
            x = torch.cat([prefix, x], dim=1)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        dropout_rate: float = 0.2,
        use_prefix: bool = False,
        use_domain_bn: bool = False,
    ) -> None:
        super().__init__()
        self.use_prefix = use_prefix
        self.base = BaseBlock(n_input, 512, use_prefix)
        self.use_domain_bn = use_domain_bn
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, n_latent)
        if self.use_domain_bn:
            self.bn_A = nn.BatchNorm1d(512)
            self.bn_B = nn.BatchNorm1d(512)
        
    def forward(self, x: torch.Tensor, prefix: Optional[torch.Tensor] = None, domain: Optional[Literal["A","B"]] = "A") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.base(x, prefix)

        if self.use_domain_bn:
            if domain == 'A':
                h = self.bn_A(h)
            elif domain == 'B':
                h = self.bn_B(h)
            else:
                raise ValueError("Domain must be 'A' or 'B'")
            
        h = F.relu(h)
        h = self.dropout(h)

        z = self.fc(h)
        return z


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        dropout_rate: float = 0.2,
        use_prefix: bool = False,
        use_domain_bn: bool = False,
    ) -> None:
        super().__init__()
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn

        self.base = BaseBlock(n_input, 512, use_prefix)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_mu = nn.Linear(512, n_latent)
        self.fc_logvar = nn.Linear(512, n_latent)

        if self.use_domain_bn:
            self.bn_A = nn.BatchNorm1d(512)
            self.bn_B = nn.BatchNorm1d(512)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        prefix: Optional[torch.Tensor] = None,
        domain: Optional[Literal["A", "B"]] = "A"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.base(x, prefix)

        if self.use_domain_bn:
            if domain == "A":
                h = self.bn_A(h)
            elif domain == "B":
                h = self.bn_B(h)
            else:
                raise ValueError("Domain must be 'A' or 'B'")

        h = F.relu(h)
        h = self.dropout(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        return z


class Generator(nn.Module):
    def __init__(
        self,
        n_latent: int,
        link_feat_num: int,
        specific_dim_A: int,
        specific_dim_B: int,
        loss_type: str = "MSE",
        use_prefix: bool = False,
        use_domain_bn: bool = False,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.use_prefix = use_prefix

        self.base = BaseBlock(n_latent, 512, use_prefix)
        self.use_domain_bn = use_domain_bn

        if self.use_domain_bn:
            self.bn_A = nn.BatchNorm1d(512)
            self.bn_B = nn.BatchNorm1d(512)

        self.fc_shared = nn.Linear(512, link_feat_num)
        self.fc_A = nn.Linear(512, specific_dim_A) if specific_dim_A > 0 else None
        self.fc_B = nn.Linear(512, specific_dim_B) if specific_dim_B > 0 else None

    def forward(self, z: torch.Tensor, prefix: Optional[torch.Tensor] = None, domain: Literal["A","B"] = "A") -> torch.Tensor:
        h = self.base(z, prefix)

        if self.use_domain_bn:
            if domain == 'A':
                h = self.bn_A(h)
            elif domain == 'B':
                h = self.bn_B(h)
            
        h = F.relu(h)
        shared_out = self.fc_shared(h)

        if domain == "A" and self.fc_A is not None:
            specific_out = self.fc_A(h)
            out = torch.cat([shared_out, specific_out], dim=-1)
        elif domain == "B" and self.fc_B is not None:
            specific_out = self.fc_B(h)
            out = torch.cat([shared_out, specific_out], dim=-1)
        else:
            out = shared_out

        if self.loss_type == "BCE":
            out = torch.sigmoid(out)
        return out


class DomainWrapper(nn.Module):
    def __init__(self, model: nn.Module, domain: Literal["A", "B"], use_prefix: bool = True) -> None:
        super().__init__()
        self.model = model
        self.use_prefix = use_prefix
        self.domain = domain
        if use_prefix:
            prefix = torch.tensor([1.0, 0.0]) if domain == "A" else torch.tensor([0.0, 1.0])
            self.register_buffer("prefix_tensor", prefix)

    def forward(self, x: torch.Tensor, prefix: Optional[torch.Tensor] = None):
        if self.use_prefix:
            batch_size = x.size(0)
            p = self.prefix_tensor.to(x.device).expand(batch_size, -1)
            return self.model(x, prefix=p, domain=self.domain)
        return self.model(x, prefix=prefix, domain=self.domain)


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
