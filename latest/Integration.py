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
import math
import os


class IntegrationModel:
    def __init__(
        self,
        adata_A: "anndata.AnnData",
        adata_B: "anndata.AnnData",
        input_key: List[Optional[str]] = ['X_pca', 'X_lsi'],
        batch_size: int = 500,
        training_steps: int = 10000,
        warmups: int = 2000,
        seed: int = 1234,
        n_latent: int = 10,
        cut_off: float = 0.9,
        lambdaRecon: float = 10.0,
        lambdaLA: float = 10.0,
        lambdaDA: float = 1.0,
        lambdaBM: float = 1.0,
        lambdamGAN: float = 1.0,
        lambdabGAN: float = 1.0,
        lambdaCLIP: float = 0.1,
        use_prior: bool = False,
        celltype_col: Optional[str] = None,
        source_col: Optional[str] = None,
        model_path = None
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.warmups = warmups
        self.n_latent = n_latent
        self.cut_off = cut_off
        self.lambdaRecon = lambdaRecon        
        self.lambdaLA = lambdaLA
        self.lambdaDA = lambdaDA
        self.lambdaBM = lambdaBM
        self.lambdamGAN = lambdamGAN
        self.lambdabGAN = lambdabGAN
        self.lambdaCLIP = lambdaCLIP
        self.use_prior = use_prior
        self.celltype_col = celltype_col
        self.source_col = source_col

        self.dataset_A = AnnDataDataset(adata_A, 
                                        input_key = input_key[0], 
                                        output_layer = None, 
                                        celltype_key=self.celltype_col, 
                                        source_key=self.source_col,
                                        mode = "integration"
                                        )
        self.dataset_B = AnnDataDataset(adata_B, 
                                        input_key = input_key[1], 
                                        output_layer = None, 
                                        celltype_key=self.celltype_col, 
                                        source_key=self.source_col,
                                        mode = "integration"
                                        )
        
        self.dataloader_A = load_data(self.dataset_A, batch_size = self.batch_size, mode = "integration")
        self.dataloader_B = load_data(self.dataset_B, batch_size = self.batch_size, mode = "integration")

        self.model_path = model_path

    def train(self) -> None:
        self._init_models_and_optimizers()
        self._set_train_mode()
        self.bm_loss = SharedBarycenterLoss(K=10, lamda_eigen=0.5, min_shared=64, warmup=self.warmups)
        iterator_A, iterator_B = iter(self.dataloader_A), iter(self.dataloader_B)
        begin_time = time.time()
        print("Training started at:", time.asctime())

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            x_A = batch_A['input'].float().to(self.device)
            x_B = batch_B['input'].float().to(self.device)

            z_A, mu_A, logvar_A = self.E_A(x_A)
            z_B, mu_B, logvar_B = self.E_B(x_B)
            sigma_A = torch.exp(0.5 * logvar_A)
            sigma_B = torch.exp(0.5 * logvar_B)

            x_AtoB = self.G_B(mu_A)
            x_BtoA = self.G_A(mu_B)

            _, mu_AtoB, _ = self.E_B(x_AtoB)
            _, mu_BtoA, _ = self.E_A(x_BtoA)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)            
            
            Sim = pairwise_cosine(mu_A.detach(), mu_B.detach())
            Sim = Sim / Sim.max()
            max_A = Sim.max(dim=1).values 
            max_B = Sim.max(dim=0).values
            mask_A = (max_A > self.cut_off)
            mask_B = (max_B > self.cut_off)

            loss_dict = {}

            # input autoencoder loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arecon - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_dict['AE'] = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_dict['LA'] = loss_LA_AtoB + loss_LA_BtoA       

            # optimal transport process
            C = pairwise_correlation_distance(batch_A['link_feat'], batch_B['link_feat']).to(self.device)
            P = unbalanced_ot(cost_pp=C, reg=0.05, reg_m=0.5, device=self.device)  

            # distribution alignment loss
            z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
            loss_dict['DA'] = torch.sum(P * z_dist) / torch.sum(P)

            # Barycenter Mapping loss
            loss_dict['BM'] = self.bm_loss(C, z_A, z_B, mask_A, mask_B, step)

            # discriminator loss
            margin = 1.0
            for _ in range(3): 
                self.optimizer_Dis_m.zero_grad() 
                loss_mDis_A = (F.softplus(-torch.clamp(self.Dis_Z(z_A[mask_A].detach()), -margin, margin))).mean()
                loss_mDis_B = (F.softplus(torch.clamp(self.Dis_Z(z_B[mask_B].detach()), -margin, margin))).mean()
                alpha = min(1.0, max(0.0, (step - self.warmups) / 500))
                loss_dict['mDis'] = alpha * (loss_mDis_A + loss_mDis_B)
                loss_dict['mDis'].backward()
                self.optimizer_Dis_m.step()

            if self.optimizer_Dis_b:
                self.optimizer_Dis_b.zero_grad()
                loss_dict['bDis'] = self.compute_discriminator_loss_intra(z_A.detach(), z_B.detach(), batch_A, batch_B)
                loss_dict['bDis'].backward()
                self.optimizer_Dis_b.step()
            else:
                loss_dict['bDis'] = torch.tensor(0.0, device=self.device)

            # generator loss
            loss_mGAN_A = -(F.softplus(-torch.clamp(self.Dis_Z(z_A[mask_A]), -margin, margin))).mean()
            loss_mGAN_B = -(F.softplus(torch.clamp(self.Dis_Z(z_B[mask_B]), -margin, margin))).mean()
            alpha = min(1.0, max(0.0, (step - self.warmups) / 500))
            loss_dict['mGAN'] = alpha * (loss_mGAN_A + loss_mGAN_B) 
            loss_dict['bGAN'] = -self.compute_discriminator_loss_intra(z_A, z_B, batch_A, batch_B)           
            
            # semi-supervised clip loss(optional)
            if self.use_prior:
                prior_matrix = build_celltype_prior(batch_A['celltype'], batch_B['celltype']).to(self.device)
                loss_dict['CLIP'] = generalized_clip_loss_stable_masked(z_A, z_B, prior_matrix)
            else:
                loss_dict['CLIP'] = torch.tensor(0.0, device=self.device)
            
            total_loss = (
                self.lambdaRecon * loss_dict['AE']
                + self.lambdaLA * loss_dict['LA']
                + self.lambdaDA * loss_dict['DA']
                + self.lambdaBM * loss_dict['BM']
                + self.lambdamGAN * loss_dict['mGAN']
                + self.lambdabGAN * loss_dict['bGAN']
                + self.lambdaCLIP * loss_dict['CLIP']
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                self.log(step, loss_dict)

        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")

        if self.model_path:
            os.makedirs(self.model_path, exist_ok=True)

            state = {
                'E_A': self.E_A.state_dict(),
                'E_B': self.E_B.state_dict(),
                'G_A': self.G_A.state_dict(),
                'G_B': self.G_B.state_dict(),
            }

            torch.save(state, os.path.join(self.model_path, "ckpt.pth"))
        

    def get_latent_representation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A(x_A)
            _, mu_B, _ = self.E_B(x_B)

        self.latent = np.concatenate((mu_A.cpu().numpy(), mu_B.cpu().numpy()), axis=0)

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")


    def get_imputation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.cat([self.dataset_A[i]['input'].float().unsqueeze(0) for i in range(len(self.dataset_A))], dim=0).to(self.device)
        x_B = torch.cat([self.dataset_B[i]['input'].float().unsqueeze(0) for i in range(len(self.dataset_B))], dim=0).to(self.device)

        with torch.no_grad():
            _, mu_A, _ = self.E_A(x_A)
            _, mu_B, _ = self.E_B(x_B)
            x_AtoB = self.G_B(mu_A)
            x_BtoA = self.G_A(mu_B)
            self.imputed_BtoA = x_BtoA.cpu().numpy()
            self.imputed_AtoB = x_AtoB.cpu().numpy()

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")

    def compute_discriminator_loss_inter(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        return F.softplus(-self.Dis_Z(z_A)).mean() + F.softplus(self.Dis_Z(z_B)).mean()

    def compute_discriminator_loss_intra(self, z_A, z_B, batch_A, batch_B):
        losses = []
        if self.Dis_A:
            losses.append(F.cross_entropy(self.Dis_A(z_A), batch_A['source'].to(self.device)))
        if self.Dis_B:
            losses.append(F.cross_entropy(self.Dis_B(z_B), batch_B['source'].to(self.device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return sum(losses)

    def _init_models_and_optimizers(self) -> None:
        self.E_A = VAEEncoder(
            n_input=self.dataset_A.feature_shapes['input'],
            n_latent=self.n_latent
        ).to(self.device)

        self.E_B = VAEEncoder(
            n_input=self.dataset_B.feature_shapes['input'],
            n_latent=self.n_latent
        ).to(self.device)

        self.G_A = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_A.feature_shapes['input']
        ).to(self.device)

        self.G_B = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_B.feature_shapes['input']
        ).to(self.device)

        self.params_G = chain(
            self.E_A.parameters(),
            self.E_B.parameters(),
            self.G_A.parameters(),
            self.G_B.parameters()
        )

        self.optimizer_G = optim.AdamW(self.params_G, lr=1e-3, weight_decay=1e-3)

        self.Dis_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.optimizer_Dis_m = optim.AdamW(self.Dis_Z.parameters(), lr=1e-3, weight_decay=1e-3)

        self.Dis_A = (
            MultiClassDiscriminator(self.n_latent, self.dataset_A.source_categories).to(self.device)
            if self.dataset_A.source_categories > 1 else None
        )
        self.Dis_B = (
            MultiClassDiscriminator(self.n_latent, self.dataset_B.source_categories).to(self.device)
            if self.dataset_B.source_categories > 1 else None
        )

        params_Dis_b = []
        if self.Dis_A:
            params_Dis_b.extend(self.Dis_A.parameters())
        if self.Dis_B:
            params_Dis_b.extend(self.Dis_B.parameters())
        self.optimizer_Dis_b = optim.AdamW(params_Dis_b, lr=1e-3, weight_decay=1e-3) if params_Dis_b else None


    def _set_train_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.Dis_Z, self.Dis_A, self.Dis_B]:
            if model is not None:
                model.train()

    def _set_eval_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.Dis_Z, self.Dis_A, self.Dis_B]:
            if model is not None:
                model.eval()

    def log(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        print(
            f"Step {step} | "
            f"loss_Recon: {self.lambdaRecon * loss_dict['AE']:.4f} | "
            f"loss_LA: {self.lambdaLA * loss_dict['LA']:.4f} | "
            f"loss_DA: {self.lambdaDA * loss_dict['DA']:.4f} | "
            f"loss_BM: {self.lambdaBM * loss_dict['BM']:.4f} | "
            f"loss_CLIP: {self.lambdaCLIP * loss_dict['CLIP']:.4f} | "
            f"loss_mGAN: {loss_dict['mGAN']:.4f}| "
            f"loss_bGAN: {loss_dict['bGAN']:.4f}"
        )



class SharedBarycenterLoss(torch.nn.Module):
    def __init__(
        self,
        K=10,
        lamda_eigen=0.5,
        min_shared=64,
        warmup=2000,
    ):
        super().__init__()
        self.K = K
        self.lamda_eigen = lamda_eigen
        self.min_shared = min_shared
        self.warmup = warmup

    def forward(self, C, z_A, z_B, mask_A, mask_B, step):
        device = z_A.device

        if step < self.warmup:
            return torch.tensor(0.0, device=device)

        if mask_A.sum() < self.min_shared or mask_B.sum() < self.min_shared:
            return torch.tensor(0.0, device=device)

        z_A_s = z_A[mask_A]
        z_B_s = z_B[mask_B]

        C_s = C[mask_A][:, mask_B]
        P_s = unbalanced_ot(cost_pp=C_s, reg=0.05, reg_m=0.5, device=device)  

        L_A = Graph_Laplacian_torch(z_A_s, nearest_neighbor=min(self.K, z_A_s.size(0) - 1))
        L_B = Graph_Laplacian_torch(z_B_s, nearest_neighbor=min(self.K, z_B_s.size(0) - 1))

        z_A_new = Transform(z_A_s, z_B_s, P_s, L_A, lamda_Eigenvalue=self.lamda_eigen)
        z_B_new = Transform(z_B_s, z_A_s, P_s.t(), L_B, lamda_Eigenvalue=self.lamda_eigen)
        loss = torch.mean((z_A_s - z_A_new) ** 2) + torch.mean((z_B_s - z_B_new) ** 2)

        return loss

