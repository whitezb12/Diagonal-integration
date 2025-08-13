import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import ot
from mycode.dataloader import *
from mycode.utils import *
from mycode.network import *
from typing import Optional, Dict, Literal


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
        lambdaOT: float = 1.0,
        lambdamGAN: float = 1.0,
        lambdabGAN: float = 1.0,
        lambdaGeo: float = 0.1,
        n_KNN: int = 30,
        mode: str = 'weak',
        use_prior: bool = False,
        alpha: float = 2,
        celltype_col: Optional[str] = None,
        source_col: Optional[str] = None,
        loss_type: Literal['MSE', 'BCE'] = 'MSE',
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
        self.lambdaOT = lambdaOT
        self.lambdaLA = lambdaLA
        self.lambdamGAN = lambdamGAN
        self.lambdabGAN = lambdabGAN
        self.lambdaGeo = lambdaGeo
        self.n_KNN = n_KNN
        self.mode = mode
        self.use_prior = use_prior
        self.alpha = alpha
        self.loss_type = loss_type

        self.dataset_A = AnnDataDataset(adata1, celltype_key=celltype_col, source_key=source_col)
        self.dataset_B = AnnDataDataset(adata2, celltype_key=celltype_col, source_key=source_col)

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

            z_A, mu_A, logvar_A = self.E_A(x_A)
            z_B, mu_B, logvar_B = self.E_B(x_B)

            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)

            z_AtoB, _, _ = self.E_B(x_AtoB)
            z_BtoA, _, _ = self.E_A(x_BtoA)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            self.optimizer_D.zero_grad()
            loss_D_m = self.compute_discriminator_loss_inter(z_A, z_B)
            loss_D_b = self.compute_discriminator_loss_intra(z_A, z_B, batch_A, batch_B)
            loss_D = self.lambdamGAN * loss_D_m + self.lambdabGAN * loss_D_b
            loss_D.backward()
            self.optimizer_D.step()

            loss_dict = {}
            loss_dict['VAE'] = self.compute_vae_loss(x_A, x_Arecon, mu_A, logvar_A) + self.compute_vae_loss(x_B, x_Brecon, mu_B, logvar_B)
            loss_dict['LA'] = self.compute_latent_align_loss(z_A, z_AtoB) + self.compute_latent_align_loss(z_B, z_BtoA)
            loss_dict['OT'], T = self.compute_ot_loss(z_A, z_B, z_AtoB, z_BtoA, batch_A, batch_B)
            loss_dict['Geo'] = self.compute_geo_loss(z_A, z_B, T)
            loss_dict['mGAN'] = self.compute_generator_loss_inter(z_A, z_B)
            loss_dict['bGAN'] = self.compute_generator_loss_intra(z_A, z_B, batch_A, batch_B)
            total_loss = (
                self.lambdaRecon * loss_dict['VAE']
                + self.lambdaLA * loss_dict['LA']
                + self.lambdaOT * loss_dict['OT']
                + self.lambdaGeo * loss_dict['Geo']
                + self.lambdamGAN * loss_dict['mGAN']
                + self.lambdabGAN * loss_dict['bGAN']
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 100 == 0:
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
        return recon_loss + 0.01*kl_div

    def compute_latent_align_loss(self, z: torch.Tensor, z_to: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z, z_to)

    def compute_discriminator_loss_inter(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        return F.softplus(-self.D_Z(z_A.detach())).mean() + F.softplus(self.D_Z(z_B.detach())).mean()

    def compute_discriminator_loss_intra(self, z_A: torch.Tensor, z_B: torch.Tensor, batch_A: dict, batch_B: dict) -> torch.Tensor:
        loss = 0.0
        if self.D_A:
            loss += F.cross_entropy(self.D_A(z_A.detach()), batch_A['source'].to(self.device))
        if self.D_B:
            loss += F.cross_entropy(self.D_B(z_B.detach()), batch_B['source'].to(self.device))
        return loss

    def compute_generator_loss_inter(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        return -(F.softplus(-self.D_Z(z_A)) + F.softplus(self.D_Z(z_B))).mean()

    def compute_generator_loss_intra(self, z_A: torch.Tensor, z_B: torch.Tensor, batch_A: dict, batch_B: dict) -> torch.Tensor:
        loss = 0.0
        if self.D_A:
            loss += -F.cross_entropy(self.D_A(z_A), batch_A['source'].to(self.device))
        if self.D_B:
            loss += -F.cross_entropy(self.D_B(z_B), batch_B['source'].to(self.device))
        return loss

    def compute_ot_loss(self, z_A: torch.Tensor, z_B: torch.Tensor, z_AtoB: torch.Tensor, z_BtoA: torch.Tensor, batch_A: dict, batch_B: dict) -> torch.Tensor:
        if 'link_feat' in batch_A and 'link_feat' in batch_B and self.mode == 'weak':
            c_cross = pairwise_correlation_distance(batch_A['link_feat'], batch_B['link_feat']).to(self.device)
        elif self.mode == 'strong':
            c_cross =  (pairwise_correlation_distance(z_A.detach(), z_BtoA.detach()) + pairwise_correlation_distance(z_B.detach(), z_AtoB.detach()))/2
        else:
            raise ValueError("Invalid mode for distance computation")

        if 'celltype' in batch_A and 'celltype' in batch_B and self.use_prior:
            prior_matrix = build_celltype_prior(batch_A['celltype'], batch_B['celltype'], prior=self.alpha).to(self.device)
        else:
            prior_matrix = build_mnn_prior(c_cross, self.n_KNN, prior=self.alpha)

        T = unbalanced_ot(cost_pp=c_cross, prior=prior_matrix, reg=0.05, reg_m=0.5, device=self.device)
        z_dist = torch.cdist(z_A, z_B, p=2).pow(2)
        ot_loss = torch.sum(T * z_dist) / torch.sum(T) + self.sliced_wasserstein_distance(z_A, z_B)
        return ot_loss, T
    
    def compute_geo_loss(self, z_A: torch.Tensor, z_B: torch.Tensor, T: torch.Tensor, k: int = 10) -> torch.Tensor:
        T_ab = T / (T.sum(dim=1, keepdim=True) + 1e-12)
        T_ba = T.T / (T.T.sum(dim=1, keepdim=True) + 1e-12)
        W_A = Graph_topk(z_A, nearest_neighbor=min(k, z_A.size(0)-1))
        W_B = Graph_topk(z_B, nearest_neighbor=min(k, z_B.size(0)-1))
        z_A_mapped = torch.matmul(T_ba, z_A)
        z_dist_A = torch.cdist(z_A_mapped, z_A_mapped, p=2).pow(2)
        loss_A = torch.sum(z_dist_A * W_A) / torch.sum(W_A)
        z_B_mapped = torch.matmul(T_ab, z_B)
        z_dist_B = torch.cdist(z_B_mapped, z_B_mapped, p=2).pow(2)
        loss_B = torch.sum(z_dist_B * W_B) / torch.sum(W_B)
        return (loss_A + loss_B) / 2
    
    def sliced_wasserstein_distance(self, z_A: torch.Tensor, z_B: torch.Tensor, num_projections: int = 50, p: int = 2) -> torch.Tensor:
        projections = torch.randn((num_projections, self.n_latent), device=self.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)
        proj_A = z_A @ projections.T
        proj_B = z_B @ projections.T
        proj_A_sorted, _ = torch.sort(proj_A, dim=0)
        proj_B_sorted, _ = torch.sort(proj_B, dim=0)
        distances = (proj_A_sorted - proj_B_sorted).abs().pow(p).mean(dim=0)
        return distances.mean().pow(1 / p)

    def _init_models_and_optimizers(self) -> None:
        if self.mode == 'strong':
            self.shared_encoder = Encoder(
                self.dataset_A.feature_shapes['expression'], self.n_latent, use_prefix=True, use_domain_bn=True
            ).to(self.device)
            self.E_A = DomainWrapper(self.shared_encoder, domain='A', use_prefix=True, use_domain_bn=True)
            self.E_B = DomainWrapper(self.shared_encoder, domain='B', use_prefix=True, use_domain_bn=True)
            self.shared_decoder = Generator(
                self.dataset_A.feature_shapes['expression'],
                self.n_latent,
                loss_type=self.loss_type,
                use_prefix=True,
                use_domain_bn=True,
            ).to(self.device)
            self.G_A = DomainWrapper(self.shared_decoder, domain='A', use_prefix=True, use_domain_bn=True)
            self.G_B = DomainWrapper(self.shared_decoder, domain='B', use_prefix=True, use_domain_bn=True)
            self.params_G = list(self.shared_encoder.parameters()) + list(self.shared_decoder.parameters())
        else:
            self.E_A = Encoder(
                self.dataset_A.feature_shapes['expression'], self.n_latent, use_prefix=False, use_domain_bn=False
            ).to(self.device)
            self.E_B = Encoder(
                self.dataset_B.feature_shapes['expression'], self.n_latent, use_prefix=False, use_domain_bn=False
            ).to(self.device)
            self.G_A = Generator(
                self.dataset_A.feature_shapes['expression'],
                self.n_latent,
                loss_type=self.loss_type,
                use_prefix=False,
                use_domain_bn=False,
            ).to(self.device)
            self.G_B = Generator(
                self.dataset_B.feature_shapes['expression'],
                self.n_latent,
                loss_type=self.loss_type,
                use_prefix=False,
                use_domain_bn=False,
            ).to(self.device)
            self.params_G = (
                list(self.E_A.parameters())
                + list(self.E_B.parameters())
                + list(self.G_A.parameters())
                + list(self.G_B.parameters())
            )

        self.optimizer_G = optim.AdamW(self.params_G, lr=0.001, weight_decay=0.0)

        self.D_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.D_A = (
            MultiClassDiscriminator(self.n_latent, self.dataset_A.source_categories).to(self.device)
            if self.dataset_A.source_categories > 1
            else None
        )
        self.D_B = (
            MultiClassDiscriminator(self.n_latent, self.dataset_B.source_categories).to(self.device)
            if self.dataset_B.source_categories > 1
            else None
        )

        self.params_D = list(self.D_Z.parameters())
        if self.D_A:
            self.params_D += list(self.D_A.parameters())
        if self.D_B:
            self.params_D += list(self.D_B.parameters())
        self.optimizer_D = optim.AdamW(self.params_D, lr=0.001, weight_decay=0.0)

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
            f"Recon: {self.lambdaRecon * loss_dict['VAE']:.4f} | "
            f"LA: {self.lambdaLA * loss_dict['LA']:.4f} | "
            f"OT: {self.lambdaOT * loss_dict['OT']:.4f} | "
            f"Geo: {loss_dict['Geo']:.4f} | " 
            f"mGAN: {loss_dict['mGAN']:.4f} | "
            f"bGAN: {loss_dict['bGAN']:.4f}"
        )
