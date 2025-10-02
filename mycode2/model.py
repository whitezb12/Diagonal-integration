import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mycode.dataloader import *
from mycode.utils import *
from mycode.network import *
from typing import Optional, Dict, Literal
from itertools import chain


class Model(object):
    def __init__(
        self,
        adata1,
        adata2,
        batch_size: int = 500,
        training_steps: int = 10000,
        seed: int = 1234,
        n_latent: int = 16,
        lambdaRecon: float = 10.0,
        lambdaLA: float = 10.0,
        lambdaAlign: float = 1.0,
        lambdamGAN: float = 1.0,
        lambdabGAN: float = 1.0,
        lambdaGeo: float = 0.5,
        n_KNN: int = 30,
        mode: str = 'weak',
        use_prior: bool = False,
        alpha: float = 2,
        celltype_col: Optional[str] = None,
        source_col: Optional[str] = None,
        loss_type: Literal['MSE', 'BCE'] = 'MSE',
        link_feat_num: int = None
    ) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaRecon = lambdaRecon
        self.lambdaAlign = lambdaAlign
        self.lambdaLA = lambdaLA
        self.lambdamGAN = lambdamGAN
        self.lambdabGAN = lambdabGAN
        self.lambdaGeo = lambdaGeo
        self.n_KNN = n_KNN
        self.mode = mode
        self.use_prior = use_prior
        self.alpha = alpha
        self.loss_type = loss_type
        self.celltype_col = celltype_col
        self.source_col = source_col
        self.link_feat_num = link_feat_num

        self.dataset_A = AnnDataDataset(adata1, celltype_key=self.celltype_col, source_key=self.source_col)
        self.dataset_B = AnnDataDataset(adata2, celltype_key=self.celltype_col, source_key=self.source_col)

        self.dataloader_A = load_data(self.dataset_A, self.batch_size)
        self.dataloader_B = load_data(self.dataset_B, self.batch_size)

    def train(self) -> None:
        self._init_models_and_optimizers()
        self._set_train_mode()
        iterator_A, iterator_B = iter(self.dataloader_A), iter(self.dataloader_B)
        begin_time = time.time()
        print("Training started at:", time.asctime())

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            x_A = batch_A['expression'].float().to(self.device)
            x_B = batch_B['expression'].float().to(self.device)

            if self.mode == 'strong':
                z_A, mu_A, logvar_A = self.E_A(x_A[:, :self.link_feat_num])
                z_B, mu_B, logvar_B = self.E_B(x_B[:, :self.link_feat_num])
            else:
                z_A, mu_A, logvar_A = self.E_A(x_A)
                z_B, mu_B, logvar_B = self.E_B(x_B)

            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)

            if self.mode == 'strong':
                z_AtoB, _, _ = self.E_B(x_AtoB[:, :self.link_feat_num])
                z_BtoA, _, _ = self.E_A(x_BtoA[:, :self.link_feat_num])
            else:
                z_AtoB, _, _ = self.E_B(x_AtoB)
                z_BtoA, _, _ = self.E_A(x_BtoA)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            loss_dict = {}
            n_iter = 1 if self.mode == 'strong' else 3
            for _ in range(n_iter):
                self.optimizer_D_m.zero_grad()
                loss_dict['mDis'] = self.compute_discriminator_loss_inter(z_A.detach(), z_B.detach())
                loss_dict['mDis'].backward()
                self.optimizer_D_m.step()

            if self.optimizer_D_b:
                self.optimizer_D_b.zero_grad()
                loss_dict['bDis'] = self.compute_discriminator_loss_intra(z_A.detach(), z_B.detach(), batch_A, batch_B)
                loss_dict['bDis'].backward()
                self.optimizer_D_b.step()
            else:
                loss_dict['bDis'] = torch.tensor(0.0, device=self.device)

            loss_dict['VAE'] = self.compute_vae_loss(x_A, x_Arecon, mu_A, logvar_A) + \
                               self.compute_vae_loss(x_B, x_Brecon, mu_B, logvar_B)
            loss_dict['LA'] = self.compute_latent_align_loss(z_A, z_AtoB) + \
                              self.compute_latent_align_loss(z_B, z_BtoA)
            loss_dict['Align'], loss_dict['Geo'] = self.compute_ot_loss(z_A, z_B, batch_A, batch_B)
            loss_dict['mGAN'] = -self.compute_discriminator_loss_inter(z_A, z_B)
            loss_dict['bGAN'] = -self.compute_discriminator_loss_intra(z_A, z_B, batch_A, batch_B)

            total_loss = (
                self.lambdaRecon * loss_dict['VAE']
                + self.lambdaLA * loss_dict['LA']
                + self.lambdaAlign * loss_dict['Align']
                + self.lambdaGeo * loss_dict['Geo']
                + self.lambdamGAN * loss_dict['mGAN']
                + self.lambdabGAN * loss_dict['bGAN']
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                self.log(step, loss_dict)

        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")

    def eval(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Evaluation started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.stack([self.dataset_A[i]['expression'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['expression'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            if self.mode == 'strong':
                z_A, _, _ = self.E_A(x_A[:, :self.link_feat_num])
                z_B, _, _ = self.E_B(x_B[:, :self.link_feat_num])
            else:
                z_A, _, _ = self.E_A(x_A)
                z_B, _, _ = self.E_B(x_B)

        self.latent = np.concatenate((z_A.cpu().numpy(), z_B.cpu().numpy()), axis=0)

        end_time = time.time()
        print(f"Evaluation completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total evaluation time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")

    def compute_vae_loss(self, x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'MSE':
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        elif self.loss_type == 'BCE':
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.005 * kl_div

    def compute_latent_align_loss(self, z: torch.Tensor, z_to: torch.Tensor) -> torch.Tensor:
        return torch.mean((z - z_to) ** 2)

    def compute_discriminator_loss_inter(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        return F.softplus(-self.D_Z(z_A)).mean() + F.softplus(self.D_Z(z_B)).mean()

    def compute_discriminator_loss_intra(self, z_A, z_B, batch_A, batch_B):
        losses = []
        if self.D_A:
            losses.append(F.cross_entropy(self.D_A(z_A), batch_A['source'].to(self.device)))
        if self.D_B:
            losses.append(F.cross_entropy(self.D_B(z_B), batch_B['source'].to(self.device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return sum(losses)

    def compute_ot_loss(self, z_A: torch.Tensor, z_B: torch.Tensor, batch_A: dict, batch_B: dict) -> torch.Tensor:
        if 'link_feat' in batch_A and 'link_feat' in batch_B and self.mode == 'weak':
            c_cross = pairwise_correlation_distance(batch_A['link_feat'], batch_B['link_feat']).to(self.device)
        elif self.mode == 'strong':
            c_cross = pairwise_correlation_distance(z_A.detach(), z_B.detach())
        else:
            raise ValueError("Invalid mode for distance computation")

        if self.celltype_col in batch_A and self.celltype_col in batch_B and self.use_prior:
            prior_matrix = build_celltype_prior(batch_A[self.celltype_col], batch_B[self.celltype_col], prior=self.alpha).to(self.device)
        else:
            prior_matrix = build_mnn_prior(c_cross, self.n_KNN, prior=self.alpha)

        T = unbalanced_ot(cost_pp=c_cross, prior=prior_matrix, reg=0.05, reg_m=0.5, device=self.device)
        L_A = Graph_Laplacian_torch(z_A, nearest_neighbor=min(30, z_A.size(0)-1))
        L_B = Graph_Laplacian_torch(z_B, nearest_neighbor=min(30, z_B.size(0)-1))
        z_A_new = Transform(z_A, z_B, T, L_A, lamda_Eigenvalue=0.5)
        z_B_new = Transform(z_B, z_A, torch.t(T), L_B, lamda_Eigenvalue=0.5)
        z_dist = pairwise_euclidean_distance(z_A, z_B, clip=False)
        loss_geo = torch.mean((z_B - z_B_new) ** 2) + torch.mean((z_A - z_A_new) ** 2)
        loss_align = torch.sum(T * z_dist) / torch.sum(T)
        return loss_align, loss_geo

    def _init_models_and_optimizers(self) -> None:
        if self.mode == 'strong':
            self.encoder = Encoder(
                n_input = self.link_feat_num,
                n_latent = self.n_latent,
                use_prefix = True,
                use_domain_bn = False
            ).to(self.device)

            self.E_A = DomainWrapper(self.encoder, domain='A', use_prefix=True)
            self.E_B = DomainWrapper(self.encoder, domain='B', use_prefix=True)

            specific_dim_A = self.dataset_A.feature_shapes['expression'] - self.link_feat_num
            specific_dim_B = self.dataset_B.feature_shapes['expression'] - self.link_feat_num
            assert specific_dim_A >= 0 and specific_dim_B >= 0, \
                "link_feat_num must be <= feature_shapes['expression']"

            self.decoder = Generator(
                n_latent = self.n_latent,
                link_feat_num = self.link_feat_num,
                specific_dim_A = specific_dim_A,
                specific_dim_B = specific_dim_B,
                loss_type = self.loss_type,
                use_prefix = True,
                use_domain_bn= False
            ).to(self.device)

            self.G_A = DomainWrapper(self.decoder, domain='A', use_prefix=True)
            self.G_B = DomainWrapper(self.decoder, domain='B', use_prefix=True)

            self.params_G = chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
            )
        else:
            self.E_A = Encoder(
                n_input = self.dataset_A.feature_shapes['expression'],
                n_latent = self.n_latent,
                use_prefix = False,
                use_domain_bn = False 
            ).to(self.device)
            self.E_B = Encoder(
                n_input = self.dataset_B.feature_shapes['expression'],
                n_latent = self.n_latent,
                use_prefix = False,
                use_domain_bn = False 
            ).to(self.device)
            self.G_A = Generator(
                n_latent = self.n_latent,
                link_feat_num = self.dataset_A.feature_shapes['expression'],
                specific_dim_A = 0,
                specific_dim_B = 0,
                loss_type = self.loss_type,
                use_prefix = False,
                use_domain_bn = False 
            ).to(self.device)
            self.G_B = Generator(
                n_latent = self.n_latent,
                link_feat_num = self.dataset_B.feature_shapes['expression'],
                specific_dim_A = 0,
                specific_dim_B = 0,
                loss_type = self.loss_type,
                use_prefix = False,
                use_domain_bn = False 
            ).to(self.device)

            self.params_G = chain(
                self.E_A.parameters(),
                self.E_B.parameters(),
                self.G_A.parameters(),
                self.G_B.parameters()
            )

        self.optimizer_G = optim.AdamW(self.params_G, lr=1e-3, weight_decay=1e-3)
        self.D_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.optimizer_D_m = optim.AdamW(self.D_Z.parameters(), lr=1e-3, weight_decay=1e-3)

        self.D_A: Optional[nn.Module] = (
            MultiClassDiscriminator(self.n_latent, self.dataset_A.source_categories).to(self.device)
            if self.dataset_A.source_categories > 1 else None
        )
        self.D_B: Optional[nn.Module] = (
            MultiClassDiscriminator(self.n_latent, self.dataset_B.source_categories).to(self.device)
            if self.dataset_B.source_categories > 1 else None
        )

        params_D_b = []
        if self.D_A:
            params_D_b.extend(self.D_A.parameters())
        if self.D_B:
            params_D_b.extend(self.D_B.parameters())
        self.optimizer_D_b = optim.AdamW(params_D_b, lr=1e-3, weight_decay=1e-3) if params_D_b else None


    def _set_train_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_Z, self.D_A, self.D_B]:
            if model is not None:
                model.train()

    def _set_eval_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_Z, self.D_A, self.D_B]:
            if model is not None:
                model.eval()

    def log(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        print(
            f"Step {step} | "
            f"loss_Recon: {self.lambdaRecon * loss_dict['VAE']:.4f} | "
            f"loss_LA: {self.lambdaLA * loss_dict['LA']:.4f} | "
            f"loss_Align: {self.lambdaAlign * loss_dict['Align']:.4f} | "
            f"loss_Geo: {self.lambdaGeo * loss_dict['Geo']:.4f} | "
            f"loss_mGAN vs loss_mDis: {loss_dict['mGAN']:.4f} vs {loss_dict['mDis']:.4f} | "
            f"loss_bGAN vs loss_bDis: {loss_dict['bGAN']:.4f} vs {loss_dict['bDis']:.4f}"
        )
