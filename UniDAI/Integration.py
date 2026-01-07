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
        stage1_steps: int = 2000,
        stage2_steps: int = 3000,
        stage3_steps: int = 10000,
        seed: int = 1234,
        n_latent: int = 10,
        lambdaRecon: float = 10.0,
        lambdaLA: float = 10.0,
        lambdaDA: float = 1.0,
        lambdamGAN: float = 1.0,
        lambdabGAN: float = 1.0,
        lambdaCLIP: float = 0.1,
        use_prior: bool = False,
        use_mask: bool = False,
        celltype_col: Optional[str] = None,
        source_col: Optional[str] = None,
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.stage3_steps = stage3_steps
        self.n_latent = n_latent
        self.lambdaRecon = lambdaRecon        
        self.lambdaLA = lambdaLA
        self.lambdaDA = lambdaDA
        self.lambdamGAN = lambdamGAN
        self.lambdabGAN = lambdabGAN
        self.lambdaCLIP = lambdaCLIP        
        self.use_prior = use_prior
        self.use_mask = use_mask
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

        self.is_shared_A = torch.ones(adata_A.shape[0], dtype=torch.bool, device=self.device)
        self.is_shared_B = torch.ones(adata_B.shape[0], dtype=torch.bool, device=self.device)

    
    def train(self):

        begin_time = time.time()
        print("Training started at:", time.asctime())

        self._train_stage1(self.stage1_steps)
        self._train_stage2(self.stage2_steps)
        if self.use_mask:
            self.update_shared_mask()
        self._train_stage3(self.stage3_steps)
        
        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")

    
    def _train_stage1(self, training_steps):

        self._init_models_and_optimizers()
        self._init_low_encoders()
        self._set_train_mode()

        print("===== Stage 1: Initialize Matching =====")

        iterator_A = iter(self.dataloader_A)
        iterator_B = iter(self.dataloader_B)

        for step in range(training_steps+1):

            batch_A = next(iterator_A)
            batch_B = next(iterator_B)

            x_A = batch_A['input'].float().to(self.device)
            x_B = batch_B['input'].float().to(self.device)

            z_A, mu_A, logvar_A = self.E_A_fast(x_A)
            z_B, mu_B, logvar_B = self.E_B_fast(x_B)

            sigma_A = torch.exp(0.5 * logvar_A)
            sigma_B = torch.exp(0.5 * logvar_B)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
            _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))

            # input autoencoder loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arecon - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_LA = loss_LA_AtoB + loss_LA_BtoA 

            # optimal transport process
            C = pairwise_correlation_distance(batch_A['link_feat'], batch_B['link_feat']).to(self.device)
            P = unbalanced_ot(C, reg=0.05, reg_m=0.1, device=self.device)

            # distribution alignment 
            z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
            loss_DA = torch.sum(P * z_dist) / torch.sum(P)
            
            total_loss = (
                self.lambdaRecon * loss_AE
                + self.lambdaLA * loss_LA
                + self.lambdaDA * loss_DA
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 500 == 0:
                print(f"[Stage1 {step}] AE: {loss_AE:.4f} | LA: {loss_LA:.4f} | DA: {loss_DA:.4f} ")
        
        self._hard_update(self.E_A_fast, self.E_A_slow)
        self._hard_update(self.E_B_fast, self.E_B_slow)    

    def _train_stage2(self, training_steps):

        self._set_train_mode()

        print("===== Stage 2: Iterative Alignment =====")

        iterator_A = iter(self.dataloader_A)
        iterator_B = iter(self.dataloader_B)

        for step in range(training_steps+1):

            batch_A = next(iterator_A)
            batch_B = next(iterator_B)

            x_A = batch_A['input'].float().to(self.device)
            x_B = batch_B['input'].float().to(self.device)

            _, link_A, _ = self.E_A_slow(x_A)
            _, link_B, _ = self.E_B_slow(x_B)

            z_A, mu_A, logvar_A = self.E_A_fast(x_A)
            z_B, mu_B, logvar_B = self.E_B_fast(x_B)

            sigma_A = torch.exp(0.5 * logvar_A)
            sigma_B = torch.exp(0.5 * logvar_B)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
            _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))

            # input autoencoder loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arecon - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_LA = loss_LA_AtoB + loss_LA_BtoA                     
            
            # optimal transport process
            C = pairwise_correlation_distance(link_A, link_B).to(self.device)
            P = unbalanced_ot(C, reg=0.05, reg_m=0.1, device=self.device)
            
            # distribution alignment 
            z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
            loss_DA = torch.sum(P * z_dist) / torch.sum(P)

            total_loss = (
                self.lambdaRecon * loss_AE
                + self.lambdaLA * loss_LA
                + self.lambdaDA * loss_DA
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                print(
                    f"[Stage2 {step}] "
                    f"AE: {loss_AE:.4f} | "
                    f"LA: {loss_LA:.4f} | "
                    f"DA: {loss_DA:.4f}  "
                )
                self._hard_update(self.E_A_fast, self.E_A_slow)
                self._hard_update(self.E_B_fast, self.E_B_slow)

        self._hard_update(self.E_A_fast, self.E_A_slow)
        self._hard_update(self.E_B_fast, self.E_B_slow)
        

    def _train_stage3(self, training_steps):

        self._set_train_mode()

        print("===== Stage 3: Shared Alignment =====")

        iterator_A = iter(self.dataloader_A)
        iterator_B = iter(self.dataloader_B)

        for step in range(training_steps+1):

            batch_A = next(iterator_A)
            batch_B = next(iterator_B)

            x_A = batch_A['input'].float().to(self.device)
            x_B = batch_B['input'].float().to(self.device)

            _, link_A, _ = self.E_A_slow(x_A)
            _, link_B, _ = self.E_B_slow(x_B)

            z_A, mu_A, logvar_A = self.E_A_fast(x_A)
            z_B, mu_B, logvar_B = self.E_B_fast(x_B)

            sigma_A = torch.exp(0.5 * logvar_A)
            sigma_B = torch.exp(0.5 * logvar_B)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            _, mu_AtoB, _ = self.E_B_fast(self.G_B(mu_A))
            _, mu_BtoA, _ = self.E_A_fast(self.G_A(mu_B))
            
            # shared gate 
            mask_A = self.is_shared_A[batch_A['index']]
            mask_B = self.is_shared_B[batch_B['index']]            
            
            min_shared = 100
            if mask_A.sum() < min_shared or mask_B.sum() < min_shared:
                continue

            # input autoencoder loss
            beta = 0.01
            loss_AE_A = torch.mean((x_Arecon - x_A)**2) + beta * kl_divergence(mu_A, logvar_A)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2) + beta * kl_divergence(mu_B, logvar_B)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((mu_A - mu_AtoB)**2) 
            loss_LA_BtoA = torch.mean((mu_B - mu_BtoA)**2) 
            loss_LA = loss_LA_AtoB + loss_LA_BtoA                     
            
            # optimal transport process
            C = pairwise_correlation_distance(link_A, link_B).to(self.device)
            P = unbalanced_ot(C, reg=0.05, reg_m=0.1, device=self.device)

            # distribution alignment 
            z_dist = pairwise_euclidean_distance(mu_A, mu_B) + pairwise_euclidean_distance(sigma_A, sigma_B)
            loss_DA = torch.sum(P * z_dist) / torch.sum(P)

            # discriminator loss
            for _ in range(3):
                self.optimizer_Dis_m.zero_grad() 
                loss_mDis_A = (F.softplus(-self.Dis_Z(z_A[mask_A].detach()))).mean()
                loss_mDis_B = (F.softplus(self.Dis_Z(z_B[mask_B].detach()))).mean()
                loss_mDis = loss_mDis_A + loss_mDis_B
                loss_mDis.backward()
                self.optimizer_Dis_m.step()

            if self.optimizer_Dis_b:
                self.optimizer_Dis_b.zero_grad()
                loss_bDis = self.compute_discriminator_loss_intra(z_A.detach(), z_B.detach(), batch_A, batch_B)
                loss_bDis.backward()
                self.optimizer_Dis_b.step()
            else:
                loss_bDis = torch.tensor(0.0, device=self.device)

            # generator loss
            loss_mGAN_A = -(F.softplus(-self.Dis_Z(z_A[mask_A]))).mean()
            loss_mGAN_B = -(F.softplus(self.Dis_Z(z_B[mask_B]))).mean()
            loss_mGAN = loss_mGAN_A + loss_mGAN_B
            loss_bGAN = -self.compute_discriminator_loss_intra(z_A, z_B, batch_A, batch_B)           
            
            # semi-supervised clip loss(optional)
            if self.use_prior:
                prior_matrix = build_celltype_prior(batch_A['celltype'], batch_B['celltype']).to(self.device)
                loss_CLIP = generalized_clip_loss_stable_masked(z_A, z_B, prior_matrix)
            else:
                loss_CLIP = torch.tensor(0.0, device=self.device)

            total_loss = (
                self.lambdaRecon * loss_AE
                + self.lambdaLA * loss_LA
                + self.lambdaDA * loss_DA
                + self.lambdaCLIP * loss_CLIP
                + self.lambdamGAN * loss_mGAN
                + self.lambdabGAN * loss_bGAN
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                print(
                    f"[Stage2 {step}] "
                    f"AE: {loss_AE:.4f} | "
                    f"LA: {loss_LA:.4f} | "
                    f"DA: {loss_DA:.4f} | "
                    f"CLIP: {loss_CLIP:.4f} | "
                    f"mGAN: {loss_mGAN:.4f} | "
                    f"bGAN: {loss_bGAN:.4f}"
                )
        
        self._hard_update(self.E_A_fast, self.E_A_slow)
        self._hard_update(self.E_B_fast, self.E_B_slow) 
        

    def get_latent_representation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()

        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        _, mu_A, _ = self.E_A_slow(x_A)
        _, mu_B, _ = self.E_B_slow(x_B)

        self.latent = np.concatenate((mu_A.cpu().numpy(), mu_B.cpu().numpy()), axis=0)
        end_time = time.time()

        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")
    

    def update_shared_mask(self, resolution=1.0, min_shared_frac=0.05, min_similarity=0.95) -> None:
        print("Update shared mask...")
        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        _, mu_A, _ = self.E_A_slow(x_A)
        _, mu_B, _ = self.E_B_slow(x_B)
        
        latent_A = mu_A.cpu().numpy()
        latent_B = mu_B.cpu().numpy()

        self.is_shared_A, self.is_shared_B = leiden_shared_mask(z_A=latent_A, 
                                                                z_B=latent_B, 
                                                                resolution=resolution,
                                                                min_shared_frac=min_shared_frac,
                                                                min_similarity=min_similarity,
                                                                device=self.device)

        print(f"Shared cells: A {self.is_shared_A.sum().item()}, B {self.is_shared_B.sum().item()}")


    def get_imputation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.cat([self.dataset_A[i]['input'].float().unsqueeze(0) for i in range(len(self.dataset_A))], dim=0).to(self.device)
        x_B = torch.cat([self.dataset_B[i]['input'].float().unsqueeze(0) for i in range(len(self.dataset_B))], dim=0).to(self.device)

        _, mu_A, _ = self.E_A_slow(x_A)
        _, mu_B, _ = self.E_B_slow(x_B)

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

        self.E_A_fast = VAEEncoder(
            n_input=self.dataset_A.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self.E_B_fast = VAEEncoder(
            n_input=self.dataset_B.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self.G_A = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_A.feature_shapes["input"],
        ).to(self.device)

        self.G_B = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_B.feature_shapes["input"],
        ).to(self.device)

        self.params_G = chain(
            self.E_A_fast.parameters(),
            self.E_B_fast.parameters(),
            self.G_A.parameters(),
            self.G_B.parameters(),
        )

        self.optimizer_G = optim.AdamW(
            self.params_G, lr=1e-3, weight_decay=1e-3
        )

        self.Dis_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.optimizer_Dis_m = optim.AdamW(
            self.Dis_Z.parameters(), lr=1e-3, weight_decay=1e-3
        )

        self.Dis_A = (
            MultiClassDiscriminator(
                self.n_latent, self.dataset_A.source_categories
            ).to(self.device)
            if self.dataset_A.source_categories > 1
            else None
        )

        self.Dis_B = (
            MultiClassDiscriminator(
                self.n_latent, self.dataset_B.source_categories
            ).to(self.device)
            if self.dataset_B.source_categories > 1
            else None
        )

        params_Dis_b = []
        if self.Dis_A is not None:
            params_Dis_b.extend(self.Dis_A.parameters())
        if self.Dis_B is not None:
            params_Dis_b.extend(self.Dis_B.parameters())

        self.optimizer_Dis_b = (
            optim.AdamW(params_Dis_b, lr=1e-3, weight_decay=1e-3)
            if params_Dis_b
            else None
        )

    def _init_low_encoders(self) -> None:

        self.E_A_slow = VAEEncoder(
            n_input=self.dataset_A.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self.E_B_slow = VAEEncoder(
            n_input=self.dataset_B.feature_shapes["input"],
            n_latent=self.n_latent,
        ).to(self.device)

        self._hard_update(self.E_A_fast, self.E_A_slow)
        self._hard_update(self.E_B_fast, self.E_B_slow)

        for p in self.E_A_slow.parameters():
            p.requires_grad = False
        for p in self.E_B_slow.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def _hard_update(self, fast, slow):
        slow.load_state_dict(fast.state_dict())


    def _set_train_mode(self) -> None:
        for model in [
            self.E_A_fast, self.E_B_fast,
            self.G_A, self.G_B,
            self.Dis_Z, self.Dis_A, self.Dis_B
        ]:
            if model is not None:
                model.train()


    def _set_eval_mode(self) -> None:
        for model in [
            self.E_A_fast, self.E_B_fast,
            self.G_A, self.G_B,
            self.Dis_Z, self.Dis_A, self.Dis_B
        ]:
            if model is not None:
                model.eval()
    

    def save_model(self, model_path = None) -> None:
        if model_path:
            os.makedirs(model_path, exist_ok=True)
            state = {
                'E_A': self.E_A_fast.state_dict(),
                'E_B': self.E_B_fast.state_dict(),
                'G_A': self.G_A.state_dict(),
                'G_B': self.G_B.state_dict(),
            }
            torch.save(state, os.path.join(model_path, "ckpt.pth"))


