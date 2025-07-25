import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DomainWrapper(nn.Module):
    def __init__(self, base_model, domain, use_prefix=True, use_domain_bn=True):
        super().__init__()
        assert domain in ['A', 'B']
        self.model = base_model
        self.domain = domain
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn

        if use_prefix:
            self.register_buffer('prefix_tensor', torch.tensor([1., 0.]) if domain == 'A' else torch.tensor([0., 1.]))

    def forward(self, x):
        if self.use_prefix:
            batch_size = x.size(0)
            prefix = self.prefix_tensor.expand(batch_size, 2)
            x = torch.cat([prefix, x], dim=1)

        if self.use_domain_bn:
            return self.model(x, domain=self.domain)
        else:
            return self.model(x)

        
class encoder(nn.Module):
    def __init__(self, n_input, n_latent, use_prefix=True, use_domain_bn=True):
        super().__init__()
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn
        self.n_latent = n_latent
        n_hidden = 512

        self.input_dim = n_input + 2 if use_prefix else n_input

        self.fc1 = nn.Linear(self.input_dim, n_hidden)
        if use_domain_bn:
            self.bn_A = nn.BatchNorm1d(n_hidden)
            self.bn_B = nn.BatchNorm1d(n_hidden)
        else:
            self.bn = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, x, domain=None):
        h = self.fc1(x)

        if self.use_domain_bn:
            if domain == 'A':
                h = self.bn_A(h)
            elif domain == 'B':
                h = self.bn_B(h)
            else:
                raise ValueError("Domain must be 'A' or 'B' when use_domain_bn is True")
        else:
            h = self.bn(h)

        h = F.relu(h)
        z = self.fc2(h)
        return z


class generator(nn.Module):
    def __init__(self, n_input, n_latent, loss_type='BCE', use_prefix=True, use_domain_bn=True):
        super().__init__()
        self.loss_type = loss_type
        self.use_prefix = use_prefix
        self.use_domain_bn = use_domain_bn
        n_hidden = 512

        self.input_dim = n_latent + 2 if use_prefix else n_latent

        self.fc1 = nn.Linear(self.input_dim, n_hidden)
        if use_domain_bn:
            self.bn_A = nn.BatchNorm1d(n_hidden)
            self.bn_B = nn.BatchNorm1d(n_hidden)
        else:
            self.bn = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_input)

    def forward(self, z, domain=None):
        h = self.fc1(z)

        if self.use_domain_bn:
            if domain == 'A':
                h = self.bn_A(h)
            elif domain == 'B':
                h = self.bn_B(h)
            else:
                raise ValueError("Domain must be 'A' or 'B'")
        else:
            h = self.bn(h)

        h = F.relu(h)
        x = self.fc2(h)
        if self.loss_type == 'BCE':
            x = torch.sigmoid(x)
        return x


class BinaryDiscriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        n_hidden = 512
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        score = self.fc3(h)
        return torch.clamp(score, min=-50.0, max=50.0)


class MultiClassDiscriminator(nn.Module):
    def __init__(self, n_input, num_classes):
        super().__init__()
        n_hidden = 512
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.fc3 = nn.Linear(n_hidden, num_classes)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        logits = self.fc3(h)
        return logits
